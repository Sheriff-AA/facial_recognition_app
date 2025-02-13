from django import forms
from .models import RegisteredFace



class FaceUploadForm(forms.ModelForm):
    class Meta:
        model = RegisteredFace
        fields = ['name', 'image']


