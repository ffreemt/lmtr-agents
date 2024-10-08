# lmtr-agents
[![pytest](https://github.com/ffreemt/lmtr-agents/actions/workflows/routine-tests.yml/badge.svg)](https://github.com/ffreemt/lmtr-agents/actions)[![python](https://img.shields.io/static/v1?label=python+&message=3.8%2B&color=blue)](https://www.python.org/downloads/)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![PyPI version](https://badge.fury.io/py/lmtr-agents.svg)](https://badge.fury.io/py/lmtr-agents)

Various llm agents (agent_tr, agent_ref, agent_imp, agent_comb, agent_comb, agent_comb_imp, agent_comb_imp1) for translation

## Intro
Heavily based on the templates of Andrew Ng's translation agent, intended for use in https://llmtool.dattw.eu.org 's last boxes.

## Install it

```shell
pip install lmtr-agents

```

## Use it
Example `.env`
```
LLM_MODEL=gpt-4o-mini  # or gpt-3.5-turbo
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
```

```python
from lmtr_agents import agent_tr, agent_ref, agent_imp, agent_comb, agent_comb, agent_comb_imp, agent_comb_imp1
```