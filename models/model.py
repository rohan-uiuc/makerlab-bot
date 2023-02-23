from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM


class ChatBotModel:
    def __init__(self, model_name):
        self.model_name = model_name

        if "t5" in self.model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif "openai-gpt" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def generate_response(self, input_text, **generator_args):
        input_ids = self.model.tokenizer.encode(input_text, return_tensors='pt')

        if "t5" in self.model_name:
            generated_ids = self.model.generate(input_ids, **generator_args)
        elif "openai-gpt" in self.model_name:
            generated_ids = self.model.generate(input_ids, max_length=1000,
                                                pad_token_id=self.model.config.eos_token_id,
                                                **generator_args)

        return self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
