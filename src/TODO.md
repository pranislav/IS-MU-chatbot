## RAG branch

### Basic pipeline
- [ ] fix gemma prompting (does not behave like an assistant)
- [ ] add LangSmith to see what is happening inside
- [ ] chatting (attach history to the context window)

### Advanced stuff
- [ ] Query translation
- [ ] flow engineering

---
## Fine-tuning branch (frozen)
- [ ] Implement dynamic padding (global padding to longest causes memory issue)
- [ ] Split the computations to more GPUs
- [x] Implement early stopping (instead of fixed amount of epochs)
- [x] Check if the adapter works properly in chat.py (missing keys error) | it works, the error just went away
- [ ] Refactor to enable hyperparam tuning
- [ ] Quantization issue
- [ ] Gradient checkpointing? (should help with memory issue)
- [x] Add warmup
- [x] Check if the apply_chat_template makes any difference | yes, it does
- [ ] Add system message ("role": "system") about its context (you are ISbot) - some question may be answered generally without right contexts. I also think it will help the model to reach the info learned while seeing that message (the info from Napoveda).
