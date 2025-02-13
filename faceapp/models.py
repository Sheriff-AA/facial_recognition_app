from django.db import models
import os
import numpy as np
import face_recognition
from django.utils.timezone import now
from django.core.files.storage import default_storage
from django.conf import settings



class RegisteredFace(models.Model):
    name = models.CharField(max_length=255, unique=True)
    image = models.ImageField(upload_to='faces/')    
    embedding = models.TextField(blank=True)  # Store embedding as a string

    def save(self, *args, **kwargs):
        # Rename image file to match the 'name' field while keeping the file type
        ext = os.path.splitext(self.image.name)[-1]
        self.image.name = f"{self.name}{ext}"

        super().save(*args, **kwargs)  # Save the image first

        # Generate and store face embedding
        image_path = os.path.join(settings.MEDIA_ROOT, self.image.name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            self.embedding = np.array2string(encodings[0], separator=',')
            super().save(update_fields=["embedding"])  # Save only the embedding

    def get_embedding(self):
        """Convert stored embedding text back into a NumPy array"""
        return np.fromstring(self.embedding.strip("[]"), sep=",") if self.embedding else None

    def __str__(self):
        return self.name


class RecognitionLog(models.Model):
    name = models.CharField(max_length=255)  # Recognized person's name
    # image = models.ImageField(upload_to='logs/')  # Store recognized image
    confidence = models.FloatField()  # Similarity score
    timestamp = models.DateTimeField(default=now)  # Auto timestamp

    def __str__(self):
        return f"{self.name} recognized at {self.timestamp}"
    


