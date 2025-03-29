import numpy as np
import pixeltable as pxt

# Ensure a clean slate for the demo
pxt.drop_dir('rag_demo', force=True)
pxt.create_dir('rag_demo')

base = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'
qa_url = base + 'Q-A-Rag.xlsx'
queries_t = pxt.io.import_excel('rag_demo.queries', qa_url)