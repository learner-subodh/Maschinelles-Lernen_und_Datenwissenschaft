import pyttsx3
import PyPDF2 as pdf

my_book = input("Enter exact name of the book with .pdf extension: ")
my_page = int(input("Enter Page number (Starts with 0): "))
my_voice = int(input("Whose voice do you want: \n0 for Male \n1 for Female\n"))

if isinstance(my_page, int) != True or my_page < 0:
    print("Enter valid answers :(")

book = open(my_book, 'rb')
pdf_reader = pdf.PdfFileReader(book)
pages = pdf_reader.numPages

if my_page > pages:
    print("Number of pages entered are more than the number of ages in entered text :(")

# object creation
speaker = pyttsx3.init()

# Voices
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[my_voice].id)

# Volume
volume = speaker.getProperty('volume')                        
speaker.setProperty('volume', 1.0)

# Rate
rate = speaker.getProperty('rate')
speaker.setProperty('rate', 125)

for curr_page in range(my_page, pages):
    page = pdf_reader.getPage(curr_page)
    text = page.extractText()
    speaker.say(text)
    speaker.runAndWait()