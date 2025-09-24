import base64
import numpy as np
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
import gdown

class FaceSwapProcessor:
    """Face swap processor class"""
    
    def __init__(self):
        self.app = None
        self.swapper = None
        self.is_initialized = False
    
    def initialize_face_swap(self):
        """Initialize face swap models"""
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        model_path = os.path.join(os.path.dirname(__file__), "inswapper_128.onnx")
        if not os.path.exists(model_path):
            file_id = '1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
        else:
            print("Inswapper modeli mevcut.")
        
        self.swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
        self.is_initialized = True
        
        return self.app, self.swapper
    
    def face_swap_function(self, source_image_base64, target_image_base64, display_results=True):
        """
        Kaynak resimdeki yüzü hedef resme yerleştirir (tek yönlü face swap)
        
        Args:
            source_image_base64 (str): Base64 formatında kaynak resim (yüzü alınacak resim)
            target_image_base64 (str): Base64 formatında hedef resim (yüzün yerleştirileceği resim)
            display_results (bool): Sonuçları görüntüle
        
        Returns:
            str: Base64 formatında kodlanmış sonuç resmi
        """
        if not self.is_initialized:
            self.initialize_face_swap()
        
        # Base64'ten resimleri decode et
        try:
            # Base64 string'i byte array'e çevir
            source_img_data = base64.b64decode(source_image_base64)
            target_img_data = base64.b64decode(target_image_base64)
            
            # Byte array'i numpy array'e çevir
            source_nparr = np.frombuffer(source_img_data, np.uint8)
            target_nparr = np.frombuffer(target_img_data, np.uint8)
            
            # OpenCV ile decode et
            source_img = cv2.imdecode(source_nparr, cv2.IMREAD_COLOR)
            target_img = cv2.imdecode(target_nparr, cv2.IMREAD_COLOR)
            
        except Exception as e:
            raise ValueError(f"Base64 decode hatası: {str(e)}")
        
        if source_img is None:
            raise ValueError("Kaynak resim base64'ten decode edilemedi!")
        if target_img is None:
            raise ValueError("Hedef resim base64'ten decode edilemedi!")
            
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)
        
        if len(source_faces) == 0:
            raise ValueError("Kaynak resimde yüz bulunamadı!")
        if len(target_faces) == 0:
            raise ValueError("Hedef resimde yüz bulunamadı!")
        
        # İlk yüzleri al
        source_face = source_faces[0]  
        target_face = target_faces[0]  
        
        # Face swap işlemi (kaynak yüzü hedef resme yerleştir)
        print("Face swap işlemi yapılıyor...")
        result_img = target_img.copy()
        
        # Kaynak yüzü hedef resme yerleştir
        result_img = self.swapper.get(result_img, target_face, source_face, paste_back=True)
        
        # Resmi base64 formatına çevir
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
        