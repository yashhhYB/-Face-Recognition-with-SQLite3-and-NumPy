
# Face Recognition Attendence System with SQLite3 and NumPy

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project implements a face recognition system using Python libraries. It leverages:
- **OpenCV**: For face detection and recognition tasks.
- **NumPy**: For numerical computations during feature extraction.
- **SQLite3**: For efficient storage of facial data.

This system has potential applications in areas like security access control, user identification, and personalized experiences.

## Features
- Detects faces in images or video streams.
- Recognizes faces based on pre-populated facial data stored in an SQLite3 database.
- Offers accuracy depending on the size and quality of the training dataset.

## Installation
1. **Clone this repository:**
   ```sh
   git clone https://github.com/<username>/Face-Recognition-with-SQLite3-and-NumPy.git
   cd Face-Recognition-with-SQLite3-and-NumPy
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
- Place your facial images in the `data/raw_data` directory. (Consider organizing them by name or another identifier)
- Run the script `python src/data_processing.py` (or a similar script if you have one) to pre-process the data (resizing, normalization, etc.) if needed.

### 2. Training the Model
- Run the script `python src/train_model.py` (or a similar script if you have one) to train the face recognition model on your pre-processed data.

### 3. Running the Recognition
- Run the script `python src/main.py` (or a similar script) to perform face recognition using the trained model.
- Optional: Customize `src/main.py` to specify arguments for video input, confidence thresholds, etc.

## Project Structure

face_recognition_project/
├── data/
│   ├── raw_data/         # Raw facial image files
│   └── processed_data/   # Pre-processed data files
├── src/
│   ├── __init__.py       # Makes src a Python package
│   ├── main.py           # Main script for running the project
│   ├── database.py       # Functions related to database interaction
│   ├── face_recognition.py  # Functions related to face detection and recognition
│   ├── utils.py          # Utility functions
│   └── data_processing.py # Data processing script (if needed)
│   └── train_model.py    # Model training script (if needed)
├── models/               # Trained models and related files
├── requirements.txt      # List of project dependencies
├── README.md             # Project documentation (This file)
├── LICENSE               # License for using the project
├── .gitignore            # Files to exclude from version control
└── setup.py              # Script for packaging and installing the project (Optional)
```
             +--------------+             +--------------+
             | Raw Images  |             | Pre-processed |
             +--------------+             +--------------+
                        |
                        v
             +--------------+             +--------------+
             | Feature     |             | SQLite3      |
             | Extraction  |             | Database     |
             +--------------+             +--------------+
                        |
                        v
             +--------------+             +--------------+
             | Recognition |             | Identified    |
             | Algorithm   |             | User         |
             +--------------+             +--------------+

## Dependencies
- `opencv-python`
- `numpy`
- `sqlite3`
- (List any other required libraries)

### Version Compatibility
Consider mentioning specific versions of libraries if compatibility is critical.

## Contributing
We welcome contributions to improve this project! Please refer to the `CONTRIBUTING.md` file (if you choose to create one) for guidelines.

## Security Considerations
Be mindful of security when storing facial data. Consider encryption and responsible data handling practices.

## Testing
Thoroughly test the project locally before deploying it to ensure functionality.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

This README provides a clear overview of your project, its functionalities, usage instructions, and structure, making it easy for users to understand and contribute. Remember to replace placeholders like `<username>` with your actual information.
```

### Additional Files

#### `requirements.txt`
```plaintext
opencv-python
numpy
sqlite3
```

#### `.gitignore`
```plaintext
__pycache__/
*.pyc
data/raw_data/
data/processed_data/
models/
```

#### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name='face_recognition_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'sqlite3'
    ],
    entry_points={
        'console_scripts': [
            'face_recognition=src.main:main',
        ],
    },
)
```

#### `src/__init__.py`
```python
# Initialize the src package
```

#### `src/main.py`
```python
import numpy as np
import sqlite3
import cv2
from src.database import init_db, add_face_data, get_face_data
from src.face_recognition import detect_faces, recognize_faces

def main():
    # Initialize database
    db_connection = init_db()

    # Add example face data
    example_face_data = np.random.rand(128)
    add_face_data(db_connection, 'John Doe', example_face_data)

    # Retrieve face data
    face_data = get_face_data(db_connection, 'John Doe')
    print(f"Retrieved face data for John Doe: {face_data}")

    # Perform face recognition
    recognize_faces(example_face_data, face_data)

if __name__ == "__main__":
    main()
```

#### `src/database.py`
```python
import sqlite3
import numpy as np

def init_db():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                      (id INTEGER PRIMARY KEY, name TEXT, data BLOB)''')
    conn.commit()
    return conn

def add_face_data(conn, name, data):
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO faces (name, data) VALUES (?, ?)''', (name, data.tobytes()))
    conn.commit()

def get_face_data(conn, name):
    cursor = conn.cursor()
    cursor.execute('''SELECT data FROM faces WHERE name = ?''', (name,))
    data = cursor.fetchone()
    if data:
        return np.frombuffer(data[0], dtype=np.float64)
    return None
```

#### `src/face_recognition.py`
```python
import cv2
import numpy as np

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def recognize_faces(input_face_data, known_face_data):
    distance = np.linalg.norm(input_face_data - known_face_data)
    if distance < 0.6:
        print("Face recognized!")
    else:
        print("Face not recognized.")
```

#### `src/utils.py`
```python
# Utility functions can be added here if needed
```

