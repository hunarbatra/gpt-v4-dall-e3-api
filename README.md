_This repo has an unoffical API for GPT-4V and DALL-E3. 
main.py is setup to use GPT-4V and can be extended to run it over a dataset by calling run_experiments() in main_

# Running the API:
```
cd api && npm install
node api.js
```

# Running GPT-4V over a test sample
```
python3 main.py --exp_dir={exp_subdirectory_name}
```

# To use DALL-E 3:
```python
from models import dalle3_runner

output_base64_img = dalle3_runner(prompt)
```
