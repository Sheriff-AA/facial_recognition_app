from django.shortcuts import render
import os
from django.core.files.storage import default_storage
# from deepface import DeepFace
from PIL import Image
import numpy as np
import face_recognition
from django.conf import settings


from .models import RegisteredFace, RecognitionLog
from .forms import FaceUploadForm


def preprocess_image(image_path, max_size=(800, 800)):
    """
    Resizes an image to a maximum size while maintaining aspect ratio.
    This reduces processing time and memory usage.
    """
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size)  # Resize while keeping aspect ratio
        img.save(image_path)  # Overwrite the original file
    except Exception as e:
        print(f"Error resizing image: {e}")


def register_face(request):
    """Registers a new face, ensuring only one face is present in the image."""
    if request.method == 'POST':
        form = FaceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            temp_image_path = default_storage.save(f"temp/{uploaded_image.name}", uploaded_image)
            temp_image_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

            preprocess_image(temp_image_path, max_size=(800, 800))

            # Load image and detect faces
            image = face_recognition.load_image_file(temp_image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                return render(request, 'faceapp/register_face.html', {'form': form, 'error': "No face detected. Please upload a clear image."})

            if len(face_encodings) > 1:
                return render(request, 'faceapp/register_face.html', {'form': form, 'error': "Multiple faces detected. Please upload an image with only one face."})

            # Save face (embedding is generated in model)
            form.save()
            return render(request, 'faceapp/register_success.html', {'name': form.cleaned_data['name']})

    else:
        form = FaceUploadForm()

    return render(request, 'faceapp/register_face.html', {'form': form})


def recognize_face(request):
    """Compares all faces in an uploaded image with stored face embeddings"""
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        temp_image_path = default_storage.save(f"temp/{uploaded_image.name}", uploaded_image)
        temp_image_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

        preprocess_image(temp_image_path, max_size=(800, 800))

        # Load uploaded image and detect all faces
        image = face_recognition.load_image_file(temp_image_path)
        uploaded_encodings = face_recognition.face_encodings(image)

        if not uploaded_encodings:
            return render(request, 'faceapp/match.html', {'match': False, 'message': "No faces detected."})

        registered_faces = RegisteredFace.objects.all()

        if not registered_faces:
            return render(request, 'faceapp/match.html', {'match': False, 'message': "No registered faces found."})
        
        results = []  # Store results for multiple faces

        for idx, uploaded_embedding in enumerate(uploaded_encodings):
            best_match = None
            min_distance = 0.6  # Matching threshold

            for face in registered_faces:
                stored_embedding = face.get_embedding()
                if stored_embedding is not None:
                    # Compare embeddings using Euclidean distance
                    distance = np.linalg.norm(stored_embedding - uploaded_embedding)
                    if distance < min_distance:
                        best_match = face.name
                        min_distance = distance

            if best_match:
                results.append(f"Face {idx + 1}: Matched with {best_match}")
                RecognitionLog.objects.create(name=best_match, confidence=1 - min_distance)
            else:
                results.append(f"Face {idx + 1}: No match found")

        return render(request, 'faceapp/match.html', {'match': True, 'results': results})

    return render(request, 'faceapp/recognize.html')


# def register_face(request):
#     """Registers a new face in the database"""
#     if request.method == 'POST':
#         form = FaceUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             return render(request, 'faceapp/register_success.html', {'name': form.cleaned_data['name']})
#     else:
#         form = FaceUploadForm()
#     return render(request, 'faceapp/register_face.html', {'form': form})


# def recognize_face(request):
#     """Compares uploaded face with registered faces"""
#     if request.method == 'POST' and request.FILES.get('image'):
#         uploaded_image = request.FILES['image']
#         image_path = default_storage.save(f"temp/{uploaded_image.name}", uploaded_image)
#         image_path = os.path.join(default_storage.location, image_path)

#         registered_faces = RegisteredFace.objects.all()

#         for face in registered_faces:
#             try:
#                 result = DeepFace.verify(image_path, face.image.path)
#                 if result["verified"]:
#                     return render(request, 'faceapp/match.html', {'match': True, 'name': face.name})
#             except Exception as e:
#                 print(f"Error processing face: {e}")

#         return render(request, 'faceapp/match.html', {'match': False})

#     return render(request, 'faceapp/recognize.html')


# def recognize_face(request):
#     """Compares an uploaded image with stored face embeddings"""
#     if request.method == 'POST' and request.FILES.get('image'):
#         uploaded_image = request.FILES['image']
#         temp_image_path = default_storage.save(f"temp/{uploaded_image.name}", uploaded_image)
#         temp_image_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

#         # Extract embedding from uploaded image
#         image = face_recognition.load_image_file(temp_image_path)
#         uploaded_encodings = face_recognition.face_encodings(image)

#         if uploaded_encodings:
#             uploaded_embedding = uploaded_encodings[0]
#             registered_faces = RegisteredFace.objects.all()

#             for face in registered_faces:
#                 stored_embedding = face.get_embedding()
#                 if stored_embedding is not None:
#                     # Compare embeddings using cosine similarity
#                     distance = np.linalg.norm(stored_embedding - uploaded_embedding)
#                     if distance < 0.6:  # Threshold for matching
#                         return render(request, 'faceapp/match.html', {'match': True, 'name': face.name})

#         return render(request, 'faceapp/match.html', {'match': False})

#     return render(request, 'faceapp/recognize.html')


