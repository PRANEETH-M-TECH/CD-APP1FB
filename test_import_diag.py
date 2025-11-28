try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Import from langchain_text_splitters successful")
except ImportError:
    print("Import from langchain_text_splitters failed")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("Import from langchain.text_splitter successful")
except ImportError:
    print("Import from langchain.text_splitter failed")
