from django.shortcuts import render
import os
from django.core.files.storage import default_storage
# from deepface import DeepFace
import numpy as np
import face_recognition
from django.conf import settings


from .models import RegisteredFace
from .forms import FaceUploadForm


def register_face(request):
    """Registers a new face in the database"""
    if request.method == 'POST':
        form = FaceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return render(request, 'faceapp/register_success.html', {'name': form.cleaned_data['name']})
    else:
        form = FaceUploadForm()
    return render(request, 'faceapp/register_face.html', {'form': form})


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


def recognize_face(request):
    """Compares an uploaded image with stored face embeddings"""
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        temp_image_path = default_storage.save(f"temp/{uploaded_image.name}", uploaded_image)
        temp_image_path = os.path.join(settings.MEDIA_ROOT, temp_image_path)

        # Extract embedding from uploaded image
        image = face_recognition.load_image_file(temp_image_path)
        uploaded_encodings = face_recognition.face_encodings(image)

        if uploaded_encodings:
            uploaded_embedding = uploaded_encodings[0]
            registered_faces = RegisteredFace.objects.all()

            for face in registered_faces:
                stored_embedding = face.get_embedding()
                if stored_embedding is not None:
                    # Compare embeddings using cosine similarity
                    distance = np.linalg.norm(stored_embedding - uploaded_embedding)
                    if distance < 0.6:  # Threshold for matching
                        return render(request, 'faceapp/match.html', {'match': True, 'name': face.name})

        return render(request, 'faceapp/match.html', {'match': False})

    return render(request, 'faceapp/recognize.html')


