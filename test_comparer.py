import io
from pathlib import Path
from src.document_compare.data_ingestion import DocumentIngestion
from src.document_compare.document_comparer import DocumentComparatorLLM

def load_fake_uploaded_file(file_path:Path):
    return io.BytesIO(file_path.read_bytes())

def test_compare_docuemnts():
    ref_path = Path("/Users/anujpandey/llm_projects/llmops/document_portal/data/document_compare/Long_Report_V1.pdf")
    act_path = Path("/Users/anujpandey/llm_projects/llmops/document_portal/data/document_compare/Long_Report_V2.pdf")
    
    class FakeUpload:
        def __init__(self,file_path:Path):
            self.name = file_path.name
            self._buffer =  file_path.read_bytes()

        def getbuffer(self):
           return self._buffer
       
    comparator = DocumentIngestion()
    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)
    
    ref_file, act_file = comparator.save_uploaded_files(ref_upload, act_upload)
    combined_text = comparator.combine_documents()
    
    print("\n Combined Text Preview (First 1000 chars):\n")
    print(combined_text[:1000])
    
    llm_comparator = DocumentComparatorLLM()
    comparison_df = llm_comparator.compare_documents(combined_text)
    
    print("\n=== COMPARISON RESULT ===")
    print(comparison_df.head())
    
if __name__ == "__main__":
    test_compare_docuemnts()
    