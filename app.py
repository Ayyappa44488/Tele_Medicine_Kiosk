from flask import Flask, render_template, Response,jsonify,request,session,url_for,redirect
import cv2
import face_recognition
import os
import mediapipe as mp
import base64
import numpy as np
import re
import mysql.connector
from email.message import EmailMessage
import smtplib
import ssl
import random
from googletrans import Translator
from googletrans.constants import LANGUAGES
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from spacy.lang.en import STOP_WORDS
import datetime
from gtts import gTTS
import pygame
from io import BytesIO
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key="ayyappa"
conn=mysql.connector.connect(host="localhost",user="root",password="",database="kiosk")
cursor = conn.cursor(dictionary=True) 
email_sender = 'your_mail'
email_password = 'your_password'
nltk.download('stopwords')
max_length = 128
# Choose device (cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def extract_face(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            face_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks.landmark]
            x_min, y_min = min(face_landmarks, key=lambda x: x[0])[0], min(face_landmarks, key=lambda x: x[1])[1]
            x_max, y_max = max(face_landmarks, key=lambda x: x[0])[0], max(face_landmarks, key=lambda x: x[1])[1]
            face_region = image[y_min:y_max, x_min:x_max]
            return face_region

    return None
def face_encoding(image):
    face = extract_face(image)
    if face is not None:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(face_rgb)
        if encoding:
            return encoding[0]
    return None
def compare_faces(frame):
    folder_path=r"static\uploads"
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(folder_path, file) for file in image_files]
    
    current_encoding = face_encoding(frame)
    if current_encoding is not None:
        for i, image_path in enumerate(image_paths):
            distance = face_recognition.face_distance([current_encoding], face_encoding(cv2.imread(image_path)))[0]
            print(f"Frame {i + 1} - {image_path}: Distance - {distance}")

            # You can set a threshold for matching here
            if distance < 0.48:  # Adjust the threshold as needed
                return image_path
        return False
    else:
        return "No"
def email(body,mail):
    subject = 'Mail from kiosk'
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = mail
    em['Subject'] = subject
    em.set_content(body,subtype="html")
    context = ssl.create_default_context()
    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender,mail, em.as_string())
def translate_telugu_to_english(text):
    translator = Translator()
    result = translator.translate(text, src='te', dest='en')
    return result.text
def remove_stop_words(input_text):
    tokens = input_text.split()  # Assuming input_text is a string
    stop_words = set(STOP_WORDS)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)
def get_doctor(sentence):
    save_path = r"C:\Users\Dell\Downloads\fine_tuned_bert_model_1-20240216T155820Z-001\fine_tuned_bert_model_1"
    # Load the saved model, tokenizer, and label encoder
    loaded_model = BertForSequenceClassification.from_pretrained(save_path)
    loaded_tokenizer = BertTokenizer.from_pretrained(save_path)
    loaded_le = joblib.load(f'{save_path}/label_encoder.joblib')

    # Input text for prediction
    input_text = sentence
    user_input_without_stop_words = remove_stop_words(input_text)
    if "cough" in user_input_without_stop_words or "coughing" in user_input_without_stop_words or "fever" in user_input_without_stop_words or "cold" in user_input_without_stop_words:return "General Physician"
    # Tokenize the input text
    input_encoding = loaded_tokenizer(user_input_without_stop_words, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    # Move input tensors to the device (cuda or cpu)
    # Choose device (cuda or cpu)
    input_encoding = {key: val.to(device) for key, val in input_encoding.items()}


    # Make prediction
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(**input_encoding)

    # Get predicted label
    logits = output.logits
    _, predicted_label = torch.max(logits, dim=1)

    # Convert predicted label to the original label using the label encoder
    predicted_specialization = loaded_le.inverse_transform([predicted_label.item()])
    session['predict_doctor']=predicted_specialization[0].lower()
    return predicted_specialization[0]
def get_time_slot(date,email1):
    time_slots_data = ['9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00']
    sql = "SELECT * FROM appointments WHERE email = %s and date=%s"
    condition_value = (email1,date)
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    if len(result)==0:
            return "9:00"
    else:
        for i in result:
            j=str(i['time'])
            # print(j[:-3])
            time_slots_data.remove(j[:-3])
            
        return time_slots_data[0]
def to_telugu(name):
    translator = Translator()
    result = translator.translate(name, src='en', dest='te')
    return result.text
def text_to_speech(text, language='te'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_bytes_io = BytesIO()
    tts.write_to_fp(audio_bytes_io)
    
    audio_bytes_io.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes_io)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/doctorregister')
def doctorregister():
    return render_template('doctorregister.html')
@app.route('/asha')
def asha():
    return render_template('asha.html')
@app.route('/meet')
def meet():
    if 'name' in session:
        return render_template('join.html',name=session['name'])
    return render_template('join.html')
@app.route('/registerpage')
def registerpage():
    return render_template('register.html')
@app.route('/face_recognition')
def video():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/voice')
def voice():
    sql = "SELECT * FROM appointments WHERE phone = %s and active = %s"
    condition_value = (session['user_phone'],1)
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    for i in result:
        url=i['link']
    if len(result)==0:
        telugu_text = "స్వాగతం మీరు తర్వాత పేజీ లో ఉన్న నీలం రంగు బటన్ నొక్కి సంభాషణని ప్రారంభించండి"
        text_to_speech(telugu_text)
        return render_template('voice.html')
    else:
        return redirect(url)
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Get the uploaded image file
        image_file = request.files['image']

        # Get other data from the form
        row_id = request.form['id']
        image_data = image_file.read()
        sql="insert into prescription values(%s,%s)"
        val=(row_id,image_data)
        cursor.execute(sql,val)
        conn.commit()
        sql="UPDATE appointments SET active = 0 where id=%s"
        val=(row_id,)
        cursor.execute(sql,val)
        conn.commit()
        sql="select phone from appointments where id= %s "
        val=(row_id,)
        cursor.execute(sql,val)
        phone1=cursor.fetchall()
        for i in phone1:
            phone=i['phone']
        sql="select name,address from register where phone= %s "
        val=(phone,)
        cursor.execute(sql,val)
        data=cursor.fetchall()
        for i in data:
            name=i['name']
            address=i['address']
        subject = 'Mail from kiosk'
        body="patient name is"+name+".the phone number is"+phone+"patient address is"+address+"Below is the prescription of the patient."
        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = 'sameertalagadadeevi1778@gmail.com'
        em['Subject'] = subject
        em.set_content(body, subtype='html')
        # Add the image attachment
        
        em.add_related(
            image_data,
            maintype='image',
            subtype='jpg', 
        )
        # Add SSL (layer of security)
        context = ssl.create_default_context()
        # Log in and send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, 'sameertalagadadeevi1778@gmail.com', em.as_string())
        return jsonify({'success': True, 'message': 'Image uploaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
@app.route('/signin',methods=['POST'])
def signin():
    email=request.form['email']
    password=request.form['pass1']
    sql = "SELECT * FROM doctorregister WHERE email = %s and password=%s"
    # Provide a value for the condition (replace %s with the actual value)
    sql1="SELECT * FROM asha WHERE email = %s and password=%s"
    condition_value = (email,password)
    # Execute the SQL query with the provided condition
    cursor.execute(sql, condition_value)
    # Fetch the result (in this case, a single integer representing the count)
    result = cursor.fetchall()
    for i in result:
        name1=i['name']
    cursor.execute(sql1, condition_value)
    result1=cursor.fetchall()
    for i in result1:
        name2=i['name']
    if len(result)==1:
        session['doctor']=True
        session['email']=email
        return redirect(url_for('index'))
    elif len(result1) == 1:
        session['asha']=True
        session['email']=email
        return redirect(url_for('index'))
    if email=="kchinnareddy2016@gmail.com" and password=="Ayyappa@2003":
        session['user']=True
        return redirect(url_for('index'))
    return render_template('login.html',message="Invalid Credentials")
@app.route('/doctordata',methods=['POST'])
def doctordata():
    if request.method == 'POST':
        # Access form data
        name = request.form['user']
        qualification = request.form['qualification']
        experience = request.form['experience']
        phone= request.form['phone']
        email=request.form['email']
        pass1=request.form['pass1']
        sql = "INSERT INTO doctorregister VALUES (%s, %s,%s,%s,%s,%s)"
        val = (name,qualification,experience,phone,email,pass1)
        cursor.execute(sql, val)
        conn.commit()
        return render_template('home.html')
@app.route('/ashadata',methods=['POST'])
def ashadata():
    if request.method == 'POST':
        # Access form data
        name =request.form['user']
        phone=request.form['phone']
        email=request.form['email']
        pass1=request.form['pass1']
        print(name,phone,email,pass1)
        sql = "INSERT INTO asha VALUES (%s, %s,%s,%s)"
        val = (name,phone,email,pass1)
        cursor.execute(sql, val)
        conn.commit()
        return render_template('home.html')
@app.route('/backend-endpoint', methods=['POST'])
def backend_endpoint():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract the base64-encoded frame from the data
        frame_data_url = data.get('frame', '')
        _, frame_data = frame_data_url.split(',')

        # Decode the base64 data to obtain the binary image data
        frame_binary = base64.b64decode(frame_data)

        # Convert the binary image data to a NumPy array
        frame_np = np.frombuffer(frame_binary, dtype=np.uint8)

        # Decode the NumPy array to a CV2 image
        frame_cv2 = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        matched_face_path = compare_faces(frame_cv2)
        if matched_face_path=="No":
            return jsonify({'status':'No'})
        if matched_face_path:
            print(matched_face_path)
            number=matched_face_path.split("\\")[-1].split(".")[0]
            cursor.execute("SELECT name FROM register WHERE phone=%s",(number,))
            data3 = cursor.fetchall()
            for row in data3:
                name=row['name']
            session['name']=name
            session['telugu']=to_telugu(name)
            session['user_phone']=number
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status':'failed'})
    except Exception as e:
        # Handle any exceptions that may occur during frame processing
        print('Error processing frame:', e)
        return jsonify({'status': 'error'})
@app.route('/success')
def success():
    roomID =random.randint(1000,10000)
    return render_template('success.html',name=session['name'])
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        # Access form data
        name = request.form['name']
        phone = request.form['phone']
        address = request.form['address']
        captured_photo_data = request.form['capturedPhotoData']
        # Extract the base64-encoded image data
        match = re.match(r'data:image/(\w+);base64,(.+)', captured_photo_data)
        image_data = match.group(2)

        # Decode the base64-encoded image data
        decoded_image_data = base64.b64decode(image_data)
        os.makedirs('static/uploads', exist_ok=True)
        # Save the image to a file
        file_path = f'static/uploads/{phone}.png'
        with open(file_path, 'wb') as file:
            file.write(decoded_image_data)
        sql="select * from register where phone=%s"
        val=(phone,)
        cursor.execute(sql,val)
        result=cursor.fetchall()
        if len(result)==1:
            return render_template('register.html',message="Phone number already exists")
        sql = "INSERT INTO register VALUES (%s, %s,%s)"
        val = (name,phone,address)
        cursor.execute(sql, val)
        conn.commit()
        print(f"Name: {name}, Phone: {phone}, address: {address}")
        return render_template('index.html')
@app.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.form['transcript']
        print(text)
        translated_text = translate_telugu_to_english(text)
        print(translated_text)
        session['translate_text']=translated_text
        return jsonify({'status': 'success', 'translatedText': translated_text})
    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({'status': 'error', 'message': 'Translation failed'})
@app.route('/predict')
def predict():
    predicted=get_doctor(session['translate_text'])
    print(predicted)
    sql = "SELECT * FROM doctorregister WHERE qualification = %s"
    condition_value = (predicted,)
    cursor.execute(sql, condition_value)
    # Fetch the result (in this case, a single integer representing the count)
    result = cursor.fetchall()
    if len(result)==0:
        return render_template('home.html',message="No doctor available related to that field")
    for row in result:
                email1=row['email']
    roomID =random.randint(1000,10000)
    session['roomID']=roomID
    url = f"http://127.0.0.1:5000/meet?roomID={roomID}"
    date=datetime.datetime.now()
    formatted_date = date.strftime('%Y-%m-%d')
    time=get_time_slot(formatted_date,email1)
    sql = "INSERT INTO appointments(phone,email,link,date,time,active) VALUES (%s, %s,%s,%s,%s,%s)"
    val = (session['user_phone'],email1,url,formatted_date,time,1)
    cursor.execute(sql, val)
    conn.commit()
    body=f"Hello, I am {session['name']}.this is the meeting link {url}   .and the time was{time}.please check through that"
    email(body,email1)
    
    return render_template('success.html',time=time,name=to_telugu(session['name']),doctor=to_telugu(predicted))
@app.route('/asha_data_fetch')
def asha_data_fetch():
    query = """
        SELECT
            p.id AS id,
            r.name AS name,
            r.address AS address,
            p.photo AS photo
        FROM
            prescription p
        JOIN
            appointments a ON p.id = a.id
        JOIN
            register r ON a.phone = r.phone;
    """
    cursor.execute(query)
    results = cursor.fetchall()
    for row in results:
        row['photo'] = base64.b64encode(row['photo']).decode('utf-8')
    return render_template('ashadata.html', data=results)
@app.route('/ashaschedule')
def ashaschedule():
    return render_template('ashadata.html')
@app.route('/doctorschedule')
def doctorschedule():
    sql = "SELECT id,link,date,time FROM appointments WHERE email = %s and active = %s "
    condition_value = (session['email'],1)
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    return render_template('doctordata.html',data=result)
def logout():
    session.clear()
    return render_template('home.html')
@app.route('/logout')
def logout():
    session.clear()
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
