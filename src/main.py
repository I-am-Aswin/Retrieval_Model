from retrieval import retreive

def main():
    query = input("Enter a Query : ")
    data = retreive(query)
    
    data = ". ".join(data)
    
    print(data)
    
    # print( generate_content_with_llm(data) )
    

if __name__ == '__main__':
    main()