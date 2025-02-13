from django.contrib import admin

from .models import RegisteredFace, RecognitionLog


admin.site.register(RegisteredFace)
admin.site.register(RecognitionLog)



# sherif - faceapp