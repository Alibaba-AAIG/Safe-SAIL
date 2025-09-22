from flask import Flask, request, render_template_string
import html
import json
import os
import torch
from tqdm import tqdm
from nnsight.util import fetch_attr
from sae import TopKReLUEncoder
from nnsight import LanguageModel
from transformers import AutoTokenizer
from sae import get_learned_activations
from setup import *
from neuron_db import get_neurondb, get_display_table
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

app = Flask(__name__)

import json
import html
HTML_TEMPLATE = """
<style>
  #text-container {
    font-size: 20px;
    user-select: none;
  }
  .char-span {
    cursor: pointer;
    padding: 2px 4px;
    background-color: #a0d995; /* 默认绿色高亮 */
    margin: 0 1px;
    border-radius: 3px;
    transition: background-color 0.3s ease;
    display: inline-block;
  }
  .char-span.selected {
    background-color: #87cefa; /* 选中蓝色高亮 */
  }
  #table-display {
    margin-top: 20px;
    border: 1px solid #ccc;
    min-height: 80px;
    max-width: 800px;
    padding: 10px;
    background-color: #fff;
  }
  table {
    border-collapse: collapse;
    width: 100%;
  }
  th, td {
    border: 1px solid black;
    padding: 6px 12px;
    text-align: left;
  }
</style>

<div id="text-container"></div>

<div id="table-display">
  <em>点击上面的字显示对应表格</em>
</div>

<!-- 所有表格预先隐藏 -->
{table_area}

<button onclick="window.history.back()">返回上页</button>

<script>
  (function(){
    // 待渲染的文本和对应表格ID数组，保持顺序对应
    const text = {token_list};
    const tableIds = {table_list};

    const container = document.getElementById('text-container');
    const tableDisplay = document.getElementById('table-display');

    // 将文本拆成可点击<span>
    text.forEach((char, i) => {
      const span = document.createElement('span');
      span.textContent = char;
      span.classList.add('char-span');
      span.dataset.tableId = tableIds[i];

      span.addEventListener('click', () => {
        // 先清除所有选中状态
        document.querySelectorAll('.char-span.selected').forEach(el => el.classList.remove('selected'));
        span.classList.add('selected');

        // 隐藏所有表格
        text.forEach((_, idx) => {
          const t = document.getElementById(tableIds[idx]);
          t.style.display = 'none';
        });

        // 把对应表格克隆一份放进显示区域
        const targetTable = document.getElementById(span.dataset.tableId);
        if(targetTable){
          // 克隆表格，避免多处共用
          const clone = targetTable.cloneNode(true);
          clone.style.display = '';
          // 清空显示区，插入表格
          tableDisplay.innerHTML = '';
          tableDisplay.appendChild(clone);
        }
      });
      container.appendChild(span);
    });
  })();
</script>
"""

class NeuronInference:
    def __init__(self, model_path):
        self.model, self.tokenizer = setup_source_model(model_path)
        model_paths = {
            'layer0': 'data/sae_checkpoints/ckpt_layer0.pt',
            'layer8': 'data/sae_checkpoints/ckpt_layer8.pt',
            'layer17': 'data/sae_checkpoints/ckpt_layer17.pt',
            'layer26': 'data/sae_checkpoints/ckpt_layer26.pt',
            'layer35': 'data/sae_checkpoints/ckpt_layer35.pt'
        }
        indices_paths = {
            'layer0': 'data/activation/indices/indices_layer0.pt',
            'layer8': 'data/activation/indices/indices_layer8.pt',
            'layer17': 'data/activation/indices/indices_layer17.pt',
            'layer26': 'data/activation/indices/indices_layer26.pt',
            'layer35': 'data/activation/indices/indices_layer35.pt'
        }
        quantile_paths = {
            'layer0': 'data/activation/quantiles/quantile_layer0.pt',
            'layer8': 'data/activation/quantiles/quantile_layer8.pt',
            'layer17': 'data/activation/quantiles/quantile_layer17.pt',
            'layer26': 'data/activation/quantiles/quantile_layer26.pt',
            'layer35': 'data/activation/quantiles/quantile_layer35.pt'
        }
        self.sae_encoder_list = setup_sae_encoder(model_paths)
        self.neuron_indices_list = setup_selected_neuron_indices(indices_paths)
        self.quantile_list = setup_quantiles(quantile_paths)
        self.neuron_db = get_neurondb()
        print('setup finished')

    def prompt_inference(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        encoded_prompt = self.tokenizer(prompt)
        n_new_tokens = 512
        hidden_states = []
        with self.model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
            # w_outs =[fetch_attr(model,'model.layers.'+str(i)+'.mlp.down_proj') for i in [0, 8, 17, 26, 35]]
            # for layer in range(5):
            #     layer_act = w_outs[layer].output
            #     hidden_states.append(layer_act.save())
            out = self.model.generator.output.save()
        decoded_prompt = self.tokenizer.decode(out[0][0:len(encoded_prompt['input_ids'])].cpu())
        decoded_answer = self.tokenizer.decode(out[0][len(encoded_prompt['input_ids']):].cpu())
        inputs = self.tokenizer(decoded_prompt+decoded_answer, return_tensors="pt")
        special_positions = ((inputs["input_ids"] == 151644) | (inputs["input_ids"] == 151645)).nonzero()
        inputs["attention_mask"][special_positions[:, 0], special_positions[:, 1]] = 0
        # print(inputs)
        with self.model.trace(inputs) as tracer:
            w_outs =[fetch_attr(self.model,'model.layers.'+str(i)+'.mlp.down_proj') for i in [0, 8, 17, 26, 35]]
            for layer in range(5):
                layer_act = w_outs[layer].output
                hidden_states.append(layer_act.save())
        print("Prompt: ", decoded_prompt)
        print("Generated Answer: ", decoded_answer)
        layer_ids = [0, 8, 17, 26, 35]
        records_by_token = {}
        for idx, layer in enumerate(layer_ids):
            sae_encoder = self.sae_encoder_list[idx]
            mlp_out = hidden_states[idx]
            input_feature = mlp_out.permute((1, 0, 2))
            learned_activations = get_learned_activations(sae_encoder, input_feature)
            selected_acts = learned_activations[:, 0, 0, self.neuron_indices_list[idx]] #(num_tokens, num_neurons) in this layer
            quantile_by_layer = torch.tensor(self.quantile_list[idx])
            selected_acts = selected_acts / quantile_by_layer.unsqueeze(0)
            for token_idx in range(selected_acts.size(0)):
                acts_indices = selected_acts[token_idx, :].nonzero().squeeze(1).cpu().tolist()
                neuron_indices_by_layer = torch.tensor(self.neuron_indices_list[idx])
                # print(acts_indices)
                acted_neuron_ids = neuron_indices_by_layer[acts_indices]
                values = selected_acts[token_idx, acts_indices]
                distinct_neuron_ids, distinct_values = [], []
                for neuron, value in zip(acted_neuron_ids.cpu().tolist(), values.cpu().tolist()):
                    if not neuron in distinct_neuron_ids:
                        distinct_neuron_ids.append(neuron)
                        distinct_values.append(value)
                table_records = get_display_table(self.neuron_db, layer, distinct_neuron_ids, distinct_values)
        
                if not token_idx in records_by_token:
                    records_by_token[token_idx] = table_records
                else:
                    records_by_token[token_idx].extend(table_records)
        table_code = """
        <table id="{table_id}" style="display:none;">
          <thead><tr><th>neuron_id</th><th>normalized_activation</th><th>explanation</th><th>correlation_score</th></tr></thead>
          {table_row}
        </table>
        """
        def make_display_table(token_idx, records):
            row_list = []
            records = sorted(records, key=lambda x: x[1], reverse=True)
            for record in records: # neuron_name, value, explanation, correlation_score
                neuron_name = record[0]
                value = float(record[1])
                try:
                    explanation = html.escape(record[2], quote=True)
                except:
                    explanation = ""
                correlation_score = float(record[3])
                row = f"""<tbody><tr><td>{neuron_name}</td><td>{"{:.4f}".format(value)}</td><td>{explanation}</td><td>{"{:.4f}".format(correlation_score)}</td></tr></tbody>"""
                row_list.append(row)
            table_by_token = table_code.replace('{table_id}','table'+str(token_idx)).replace('{table_row}','\n'.join(row_list))
            return table_by_token
        table_id_list = []
        token_list = []
        table_list = []
        for token_idx in records_by_token:
            table_by_token = make_display_table(token_idx, records_by_token[token_idx])
            token = inputs['input_ids'][0][token_idx]
            token_str = self.tokenizer.decode(token)
            table_id_list.append('table'+str(token_idx))
            token_list.append(html.escape(token_str, quote=True))
            table_list.append(table_by_token)
        html_code = HTML_TEMPLATE.replace('{table_area}','\n\n'.join(table_list)).replace('{table_list}', json.dumps(table_id_list, ensure_ascii=False)).replace('{token_list}', json.dumps(token_list, ensure_ascii=False))
        # html_code = HTML_TEMPLATE.replace('{token_list}', json.dumps(token_list, ensure_ascii=False))
        return html_code
# setup
neuron_client = NeuronInference(os.getenv('SOURCE_MODEL', ''))
# test inference
neuron_client.prompt_inference('今天天气如何')
        
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('your_input', '')
        return neuron_client.prompt_inference(prompt)
    else:
        # GET显示输入页
        return '''
        <html><body>
          <h2>Input Query</h2>
          <form method="post" style="font-size:18px;">
            <input type="text" name="your_input" style="width:400px; font-size:18px;" placeholder="input query" required>
            <button type="submit" style="font-size:18px;">submit</button>
          </form>
        </body></html>
        '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
