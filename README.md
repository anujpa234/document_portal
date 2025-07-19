## Commands need to follow to run project

mkdir <project_folder_name>

cd <project_folder_name>

```
code .
```

## For conda environment setup

```
conra create -p <env_name> python==3.10 -y
```

```
pip install -r requirements.txt
```

## GIT setup commands

```
git init
```

```
git add .
```

```
git commit -m "<commit message>"
```
```
git push
```

### For cloning my repo use this command

```
git clone https://github.com/anujpa234/document_portal.git
```

### Min requirement for this project
1. LLM Model ##Groq(freely), openai(paid), gemini(15 days free access), claude(paid), huggingface(freely), ollama(local setup).

2. Embedding model ##openai, hf, gemini

3. Vector Database : ## InMemory ##OnDisk ## CloudBased