from core.llm.base import BaseLLM

from core.utils.documents_processor import DocumentProcessor
from core.db.vector_db import ESGVectorDB
from llms.sorcerer_supreme import SorcererSupremeLLM

def insert_docs(folder_path):
    # folder_path = "/Users/ragz/Library/Mobile Documents/com~apple~CloudDocs/Freelance/AIROI/AI/output"
    llm = SorcererSupremeLLM(config_path="config/config.yaml")
    llm.index_documents(folder=folder_path, use_esg=True)

    print("Indexed ESG documents successfully.")

def query_db():
    """Test function to query the indexed documents."""
    print("\n--- Testing Document Query ---")
    
    # Initialize the ESGVectorDB
    esg_db = ESGVectorDB(config_path="config/config.yaml")
    
    # Test with a sample query
    test_query = "What is the effective date of GRI 302?"
    print(f"Query: {test_query}")
    
    # Query using text directly
    results = esg_db.query_text(test_query, n_results=3)
    
    # Display the results
    print("\nQuery Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        if isinstance(result, dict):
            print(f"Text: {result.get('text', 'No text')[:200]}...")
            print(f"Score: {result.get('score', 'No score')}")
        else:
            print(f"Result format: {type(result)}")
            print(f"Result content: {result}")

def test_llm():
    """Test function to check if the LLM is working properly."""
    print("\n--- Testing LLM Integration ---")
    
    # Initialize the LLM
    llm = SorcererSupremeLLM(config_path="config/config.yaml")
    
    # Test with a sample prompt
    test_prompt = "What is the effective date of GRI 302?"
    print(f"Prompt: {test_prompt}")
    
    # Generate a response
    try:
        response = llm.generate(test_prompt)
        print("\nLLM Response:")
        print(response)
    except Exception as e:
        print(f"Error generating response: {e}")

def main():
    # First, index documents if needed
    insert_docs("./test_documents")
    
    # Test querying the database
    query_db()
    
    # Test the LLM integration
    test_llm()

    #INFO: Read the below code to understand how classes interact

    # # Initialize your core components here
    # print("Starting your project...")
    # sample_document = """
    #         2. Topic disclosures
    #     Disclosure 301-1 Materials used by weight or volume
    #     The reporting organization shall report the following information:
    #     REQUIREMENTS
    #     a. Total weight or volume of materials that are used to produce and package the
    #     organization's primary products and services during the reporting period, by:
    #     i. non-renewable materials used;
    #     ii. renewable materials used.
    #     2.1 When compiling the information specified in Disclosure 301-1, the reporting organization
    #     RECOMMENDATIONS
    #     should:
    #     2.1.1 include the following material types in the calculation of total materials used:
    #     2.1.1.1raw materials, i.e., natural resources used for conversion to products or
    #     services, such as ores, minerals, and wood;
    #     2.1.1.2associated process materials, i.e., materials that are needed for the
    #     manufacturing process but are not part of the final product, such as
    #     lubricants for manufacturing machinery;
    #     2.1.1.3semi-manufactured goods or parts, including all forms of materials and
    #     components other than raw materials that are part of the final product;
    #     2.1.1.4materials for packaging purposes, including paper, cardboard and
    #     plastics;
    #     2.1.2 report, for each material type, whether it was purchased from external suppliers or
    #     sourced internally (such as by captive production and extraction activities);
    #     2.1.3 report whether these data are estimated or sourced from direct measurements;
    #     2.1.4 if estimation is required, report the methods used.
    #     Guidance for Disclosure 301-1
    #     GUIDANCE
    #     The reported usage data are to reflect the material in its original state, and not to be presented
    #     with further data manipulation, such as reporting it as 'dry weight'.
    # """

    # # Initialize the DocumentProcessor (loads configuration from config.yaml and prompt file)
    # processor = DocumentProcessor(config_path="config/config.yaml")
    
    # # Process the sample document (using 'esg' as document type)
    # enriched_chunks = processor.process_document(sample_document, doc_type="esg")
    
    # if not enriched_chunks:
    #     print("No enriched chunks produced.")
    #     return
    
    # # Extract required fields for insertion into ESGVectorDB
    # ids = [chunk["metadata"]["chunk_id"] for chunk in enriched_chunks]
    # texts = [chunk["text"] for chunk in enriched_chunks]
    # metadatas = [chunk["metadata"] for chunk in enriched_chunks]
    # embeddings = [chunk["embedding"] for chunk in enriched_chunks]
    
    # # Initialize ESGVectorDB (loads its configuration from config.yaml)
    # esg_db = ESGVectorDB(config_path="config/config.yaml")
    
    # # Insert enriched chunks into ESGVectorDB
    # esg_db.insert_documents(ids, texts, metadatas, embeddings)
    # print("Inserted documents into ESGVectorDB.")
    
    # # Query ESGVectorDB using the embedding from the first enriched chunk
    # if embeddings:
    #     query_embedding = [embeddings[0]]  # Query expects a list of embedding vectors
    #     results = esg_db.query(query_embedding, n_results=3)
    #     print("Query Results:")
    #     for result in results:
    #         print(result)
    # else:
    #     print("No embeddings available for query.")
    
    # # Clear all documents from ESGVectorDB
    # esg_db.delete_all_documents()
    # print("Cleared ESGVectorDB.")
    # # try:
    #     # response = llm.generate(prompt)
    #     # print("Response:", response)
    # # except Exception as e:
    # #     print("Error generating response:", e)
    

if __name__ == "__main__":
    main()