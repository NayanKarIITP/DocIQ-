import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
print('=== ALL MODELS ===')
for m in genai.list_models():
    if 'embed' in m.name.lower() or 'embed' in str(m.supported_generation_methods).lower():
        print(m.name, '|', m.supported_generation_methods)