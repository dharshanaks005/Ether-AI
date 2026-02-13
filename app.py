from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

#moving to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

chat_history_ids = None

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids

    user_text = request.form.get("msg")
    if not user_text:
        return "Please type something."

    
    new_input_ids = tokenizer.encode(
        user_text + tokenizer.eos_token,
        return_tensors="pt"
    ).to(device)

    if chat_history_ids is not None:
        
        if chat_history_ids.shape[-1] > 300:
            chat_history_ids = chat_history_ids[:, -300:]
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

   
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 50,  
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    bot_reply = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return bot_reply if bot_reply else "I'm thinking..."

if __name__ == "__main__":
    app.run(debug=True)