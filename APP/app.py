from flask import Flask, request, render_template, redirect, url_for, jsonify
import fitz  # PyMuPDF
import os
import joblib
from transformers import pipeline
from resumeparser import EnhancedResumeParser  # Ensure EnhancedResumeParser is correctly imported

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists

# Load the trained model and TfidfVectorizer
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
tfid_path = os.path.join(os.path.dirname(__file__), 'tfid.pkl')
model = joblib.load(model_path)
tfid = joblib.load(tfid_path)

# Set up Hugging Face API token
api_token = os.getenv('huggingface_api')  # Ensure the token is set in your environment

# Load the LLaMA model using Hugging Face Transformers
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b",
    use_auth_token=api_token
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        extracted_data = process_pdf(filepath)
        profession = predict_job_profile(extracted_data)
        extracted_data['profession'] = profession
        timeline_plan = generate_timeline_plan(profession)
        extracted_data['timeline_plan'] = timeline_plan
        return render_template('index.html', extracted_data=extracted_data)
    return 'Invalid file format. Please upload a PDF.'

@app.route('/update', methods=['POST'])
def update_data():
    updated_data = {
        'name': request.form['name'],
        'email': request.form['email'],
        'mobile_number': request.form['phone'],
        'skills': request.form['skills'].split(', '),
        'education': request.form['education'].split(', '),
        'experience': request.form['experience'].split(', '),
        'no_of_pages': request.form['no_of_pages'],
        'profession': request.form['profession'],
        'timeline_plan': request.form['timeline_plan']
    }
    return render_template('index.html', extracted_data=updated_data)

def process_pdf(filepath):
    parser = EnhancedResumeParser()
    extracted_data = parser.parse_resume(filepath)
    return extracted_data

def predict_job_profile(extracted_data):
    input_data = ' '.join(extracted_data.get('skills', [])) + ' ' + ' '.join(extracted_data.get('education', []))
    input_data_tfidf = tfid.transform([input_data])
    prediction = model.predict(input_data_tfidf)
    return prediction[0] if prediction else "Unknown"

def generate_timeline_plan(profession):
    prompt = f"Generate a 6-months study timeline for preparing to become a {profession}, detailing monthly milestones and key study topics."
    response = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7, top_k=50)
    return response[0]['generated_text'].strip() if response else "Timeline generation failed."

if __name__ == '__main__':
    app.run(debug=True)
