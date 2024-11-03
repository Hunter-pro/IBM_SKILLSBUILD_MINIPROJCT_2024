import spacy
from pyresparser import ResumeParser
import docx2txt
import PyPDF2
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')

class EnhancedResumeParser:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        self.nlp = spacy.load('en_core_web_sm')
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text using multiple methods and combine results"""
        text = ""
        
        # Method 1: PDFMiner
        try:
            text += extract_text(pdf_path) + "\n"
        except:
            pass
            
        # Method 2: PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except:
            pass
            
        return text
    
    def extract_contact_info(self, text):
        """Extract email and phone number using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None
        }
    
    def extract_name(self, text):
        """Extract name using NLP"""
        potential_names = []
    
        # Common keywords to identify the name section
        name_keywords = ['name', 'contact', 'profile']
        
        # Split text by lines and search for patterns
        lines = text.split('\n')
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # If line has keywords like 'Name' or is at the top of the resume, consider it a potential name
            if any(keyword in line.lower() for keyword in name_keywords) or len(potential_names) < 3:
                # Check if it looks like a name (2-3 words, starts with uppercase)
                if 1 < len(line.split()) <= 3 and line[0].isupper():
                    potential_names.append(line.strip())
        
        # Use SpaCy as a backup if manual detection fails
        if not potential_names:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
                    potential_names.append(ent.text)
        
        # Return the most likely candidate
        return potential_names[0] if potential_names else None
    
    def extract_education(self, text):
        """Extract education information"""
        education_keywords = ['education', 'qualification', 'degree', 'university', 'college', 'school']
        education_info = []
        
        # Split text into lines
        lines = text.split('\n')
        is_education_section = False
        
        for line in lines:
            line = line.strip()
            if any(keyword.lower() in line.lower() for keyword in education_keywords):
                is_education_section = True
                if len(line) > 5:  # Avoid section headers
                    education_info.append(line)
            elif is_education_section and len(line) > 5:
                education_info.append(line)
            elif is_education_section and len(line) < 2:
                is_education_section = False
                
        return education_info
    
    def extract_skills(self, text):
        """Extract skills using NLP and custom keyword matching"""
        # Common technical skills
        technical_skills = set([
            'python', 'java', 'javascript', 'html', 'css', 'sql', 'react',
            'angular', 'node.js', 'docker', 'kubernetes', 'aws', 'azure',
            'machine learning', 'data science', 'artificial intelligence'
        ])
        
        # Extract words and remove stopwords
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Find skills
        found_skills = set()
        for token in tokens:
            if token in technical_skills:
                found_skills.add(token)
                
        # Use spaCy for additional skill extraction
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                skill = ent.text.lower()
                if skill in technical_skills:
                    found_skills.add(skill)
                    
        return list(found_skills)
    
    def parse_resume(self, pdf_path):
        """Main method to parse resume"""
        try:
            # Get base results from pyresparser
            base_results = ResumeParser(pdf_path).get_extracted_data()
        except IOError as e:
            print(f"Error loading SpaCy model in pyresparser: {e}")
            base_results = {
                'name': None,
                'email': None,
                'mobile_number': None,
                'skills': [],
                'education': [],
                'experience': [],
                'no_of_pages': 1
            }
        
        # Extract text using enhanced method
        text = self.extract_text_from_pdf(pdf_path)
        
        # Get additional contact info
        contact_info = self.extract_contact_info(text)
        
        # Combine and enhance results
        enhanced_results = {
            'name': self.extract_name(text) or base_results.get('name'),
            'email': contact_info['email'],
            'mobile_number': contact_info['phone'] or base_results.get('mobile_number'),
            'skills': list(set(base_results.get('skills', []) + self.extract_skills(text))),
            'education': self.extract_education(text),
            'no_of_pages': base_results.get('no_of_pages', 1)
        }
        
        return enhanced_results
