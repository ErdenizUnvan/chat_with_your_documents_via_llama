from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable

import requests

# 1. PDF Yükleme ve Bölme
pdf_loader = PyPDFLoader("1728286846_the_nestle_hr_policy_pdf_2012.pdf")
documents = pdf_loader.load()

# Metinleri parçalara ayır
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#chunk_size: Bölüm boyutu (her bir bölümdeki karakter sayısı)
#chunk_overlap: Bölümler arasındaki çakışma
text_chunks = text_splitter.split_documents(documents)

# 2. Embedding Modeli (HuggingFace Sentence-Transformer)
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Vektör Veritabanı Oluştur
vector_store = Chroma.from_documents(text_chunks, embeddings)

# 3. Llama3 Sorgulama Fonksiyonu
def llama3_api_call(query):
    url = 'http://127.0.0.1:8000/ai/predict'
    query_params = {'query': query}
    response = requests.get(url, params=query_params)
    if response.status_code == 200:
        return response.json()['response']['text']
    else:
        return "Error: Unable to process the request."

# 4. Runnable Olarak LocalLLM Sınıfı
class LocalLLM(Runnable):
    def invoke(self, input: str, config=None, **kwargs):
        prompt = f"Based on the following context, answer the query:\n\nContext: {input}\n\nAnswer:"
        return llama3_api_call(prompt)

# 5. Retrieval-QA Zinciri
qa_chain = RetrievalQA.from_chain_type(
    llm=LocalLLM(),  # LocalLLM sınıfını kullan
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# 6. Sorgu İşleme
query = input("Soru: ")
result = qa_chain.invoke({"query": query})  # invoke kullanarak çağırıyoruz

#query (sorgu), embedding modeline (sentence-transformers/all-mpnet-base-v2) gönderilir.
#Bu embedding modeli, query'yi vektör formuna dönüştürür.
#Ardından Chroma vektör veritabanında bulunan tüm chunk'ların embedding'leri ile bu vektör karşılaştırılır.
#En benzer 3 chunk, Kosinüs Benzerliği (Cosine Similarity) ile bulunur.
#Retriever'dan dönen en alakalı 3 chunk birleştirilerek, prompt değişkeni icinde input degiskeni oluşturulur
#prompt tamamlanilir
#prompt, Llama3 modeline HTTP isteği üzerinden şu şekilde gönderilir:
#Answer: Llama3 modeli burada bir cevap üretmesi gerektiğini "Answer" etiketinden anlar.

# Sonuç Gösterimi
print("Answer:", result["result"])
#for i, doc in enumerate(result["source_documents"]):
#    print(f"\nSource Document {i+1}:\n{doc.page_content}")

#Cosine: Kosinüs benzerliği mesafe ölçümü olarak.
#Cosine genellikle vektörlerin benzerliklerini ölçmek için tercih edilir.
#Kosinüs benzerliği iki vektör arasındaki benzerliği ölçmek için kullanılan matematiksel bir yöntemdir
#Bu yöntem, iki vektör arasındaki açıyı hesaplar ve benzerliği bu açıya göre değerlendirir.
#İki vektör arasındaki açı ne kadar küçükse, vektörler o kadar benzerdir
#Kosinüs benzerliği, vektörlerin büyüklüğünü (uzunluğunu) değil, yönünü dikkate alır.
#Örneğin, aynı içeriğe sahip uzun ve kısa bir metin arasında benzerliği doğru şekilde ölçer.
#Kosinüs benzerliği, metin veya belgelerin benzerliğini ölçmek için çok uygundur çünkü metinler vektörlere dönüştürülür (örneğin, embedding ile).
#Kosinüs benzerliği iki cümlenin veya belgenin semantik olarak ne kadar benzediğini ölçmek için kullanılır.
