from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
import base64
import os
import tempfile

class ImagenGenerator:
    """Class for generating images using Google's Imagen model"""
    
    def __init__(self, project_id: str = "presalesdemo2024", location: str = "us-central1"):
        """Initialize the Imagen generator"""
        self.project_id = project_id
        self.location = location
        self.generation_model = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize Vertex AI and the Imagen model"""
        if not self.is_initialized:
            vertexai.init(project=self.project_id, location=self.location)
            self.generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")
            self.is_initialized = True
            print("Imagen model initialized successfully")
    
    def generate_image_from_test(
        self, 
        test_sonucu: str,
        test_adı: str,
        test_aciklamasi: str,
        gender: str,
        age: int,
        image_place: str = "The place must be relevant to the test",
        image_style: str = "The theme must be relevant to the test"
    ) -> str:
        """
        Generate an image based on test data and return as base64 string
        
        Args:
            test_sonucu: Test result text
            test_adı: Test name
            test_aciklamasi: Test description
            gender: Gender of the person
            age: Age of the person
            image_place: Place setting for the image
            image_style: Style of the image
            
        Returns:
            str: Base64 encoded image
        """
        if not self.is_initialized:
            self.initialize()
        
        # Create prompt from test data
        prompt = (
            f"Draw a realistic image for this test. Do not add text. It should represent the test result. "
            f"It should look like a stock image. Only 1 person should be in the image. "
            f"Test name: {test_adı}, Result: {test_sonucu}, Description: {test_aciklamasi}, "
            f"Gender: {gender}, Age: {age}, Place: {image_place}, Style: {image_style}"
        )
        
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Generate image
        images = self.generation_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            negative_prompt="",
            person_generation="allow_all",
            safety_filter_level="block_few",
            add_watermark=True,
        )
        
        # Save to temporary file and convert to base64
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            output_file = tmp_file.name
            images[0].save(location=output_file, include_generation_parameters=False)
        
        # Read and encode to base64
        with open(output_file, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Clean up temporary file
        try:
            os.remove(output_file)
        except:
            pass
        
        print("Image generated successfully")
        return image_base64