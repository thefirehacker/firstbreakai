# Flow of Lesson 01

## Thinking of a model file which could be safetensor or gguf file . so from thinking about a file to deploying inference a model file .

Now say we have an epic model like qwen 3 0.6B i.e a model with more than half a billion params , 600 million to be accurate. https://huggingface.co/Qwen/Qwen3-0.6B this model is available on hugging face just like code is available on github. 
you can goto hugging face and then click on Files and version you will a lot of files . Large files like .safetensor files are actual models and there will be a xet tag next to it . Huggingface is like github for models infact it is based up on git . 

In git when you commit code the code size is limited to 100mb per file. This size on file isn't really an issue as even in git there is LFS. To store large files and blobs (even binaries if you want ) You can store large files in git lfs along with normal files like your python or react files .

Now think in these terms imagine a world where huggingface doesnot exist what is required to run this model file to create a chat app like chatgt and get response from your model .

###  3 Intutions
#### A model is a stack of repeated blocks its is basically a pipeline.
#### Intuition 2 — LLMs are autoregressive: one token at a time ( this is also well written )

#### The model outputs a probability distribution over the entire vocabulary
First intution is what a model looks like it looks like this ( current part is well written)
Thirdd intution explain that when you ask a question you  get a response . Like tell me the capital of India you might get a response like Delhi. ANd this is correct however if model has a total of 150K vocab size it actually gives you prob of each of 150K tokens . You have to choose the best prob ( highest ones or some other criteria via sampling - explain this part even simpler)

look at this article on decoding https://huggingface.co/blog/mlabonne/decoding-strategies

### Qwen3 0.6B why is it the right model ?
Models can be classified as 
100-200B+ Very large models , these are the top models. There are models like Sonnet, Opus , GPT 5.5 and others used heavily, There are Os models too of this size

Then comes models 30B-100B These are normally OS models which aer very useful like Qwen3 Coder

4B-30B: Small models : These can be great even on production like my fav multimodal model Phi4-Multimodal-Instruct by Microsoft

0.3-4B : Very small Models 
Qwen3-0.6B This is our goldilocks it is just right( you can explian what is goldilocks). We will use experimenst on smaller GPT-2 124m and other variants but always come back to our goldilocks model. There is a new variant to it called https://huggingface.co/Qwen/Qwen3.5-0.8B both variants have model download in millions , new variant has 2,877,090 downlaod this month. It will be a good thought exercise to find out where is the community using this model.

I have asked reddit and X about this let me know what do you think about this ?


### Scale and anatomy of model

Now lets talk about readme.md . What are markdown files how markdown is used everywhere on discord on , github and on hf.
On Discord you can create headings and sub headings . on github a README.md file in any folder is displayed inside the folder as a readable page a very nice feature other .md files can be read or rendered on clicking on it . (tell user what md file does and what it renders ).

Talk about https://dillinger.io/ and it shows full scope of md files Like how to write with Fromatting , list , Links , Code blocks , block quotes (which look rally nice ), Table and Math

talk about how Karpathy LLM tweet on https://x.com/karpathy/status/2039805659525644595?s=20 LLM knowledge base and how these use md files 

## Ruuning the actual file 
Again we are in a world with no huggingface . So now lets open any model on hf  . we goto config.json and open  by cliking it we copy whatever is in architecture and then goto google and type transformer {name of the architecure } for 3.5 its "Qwen3_5ForConditionalGeneration" and for earlier one our goldilock model https://huggingface.co/Qwen/Qwen3-0.6B its "Qwen3ForCausalLM" you will get github file modeling_qwen3.py - huggingface/transformers and in tarnsformers doc you will get class for it class transformers.Qwen3ForCausalLM ( Explain why is this clas required) 
https://huggingface.co/docs/transformers/v5.8.0/en/model_doc/qwen3#transformers.Qwen3ForCausalLM

Now if this didnt exist we would do what we would run this in pure C our repo https://github.com/thefirehacker/QWEN3-RunLocally does this ( it has a submododule that does this https://github.com/thefirehacker/qwen3.c/tree/a068f9cabd0d180b7160fabcbec368b7642c01d3)

AI Agent: Map that class and class in our C file run.c to showcasee  to user.