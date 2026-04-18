# Week 2 Examples

## Setup

1. Install requirements:
```
pip install -r requirements.txt
```

2. Create a `.env` file and add your Tavily API key:
```
TAVILY_API_KEY=your_key_here
```

3. (Added) Switch to project Python3 (run from project root folder):
```
source venv/bin/activate
```

4. (Added) Check if you are using the right Python
```
which python3 
```

5. (Added) Run the chatbot
```
python3 agent_demo.py  
```

## Running the examples

See how retrievers work:
```
python retriever.py
```

See how query engines work:
```
python query_engine.py
```

See how function agents work:
```
python function_agent.py
```

## TODO

Open `agent_demo.py` and build a simple Claude-like chatbot agent. Modify the functions in there as needed.
