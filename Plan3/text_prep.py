import re
from nltk import tokenize

def process(data_path,filename):
    tsv_str = ".tsv"
    txt_str = ".txt"
    with open(data_path+filename+txt_str,
              "r",encoding="utf8") as f:
        data = f.readlines()

    temp = [x.replace("<\|endoftext\|>","").replace("Emet-Selch:","").replace("<|endoftext|>","").strip() for x in data]
    text_cleaned = tokenize.sent_tokenize(' '.join(temp))
    output = open(data_path+filename+tsv_str, "w", encoding="utf8")
    output.write("prompt\tcompletion\n")
    for item in text_cleaned:
        output.write(f"\t{item}\n")
    output.close()

def main():

    # setups
    data_path = "C:/Users/fyqaw/Documents/NYU/2022-2023/DS203/finalProject/MLLU_Final_Project/data/"
    data_names = ["emet_ao3",
                  "emet_dialogue",
                  "gandalf_ao3",
                  "gandalf_dialogue",
                  "hermione_ao3",
                  "hermione_dialogue"]


    for file in data_names:
        process(data_path,file)



if __name__ == '__main__':
    main()
