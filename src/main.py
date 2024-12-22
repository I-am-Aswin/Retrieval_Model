from retrieval import retreive
from generate import generate_content

def main():
    query = input("Enter a Query : ")
    data = retreive(query)
    
    print(*data, sep='\n', end='\n\n')
    data = ". ".join(data)
    
    
    print( generate_content(data, query) )
    

if __name__ == '__main__':
    main()