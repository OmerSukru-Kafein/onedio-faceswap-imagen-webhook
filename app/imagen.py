from vertexai.preview.vision_models import ImageGenerationModel
import vertexai

"""
- image_place:str (opt)
- image_style:str (opt)
- test sonucu:str
- test_adı:str
- test_aciklamasi:str
- gender:str
- age:int
"""
from pydantic import BaseModel

class test_data(BaseModel):
    """Request model for test data"""
    image_place: str = "The place must be relevant to the test"
    image_style: str = "The theme must be relevant to the test"
    test_sonucu: str
    test_adı: str
    test_aciklamasi: str
    gender: str
    age: int

image_place = "room" 
image_style = "detailed"
test_sonucu = "Inanılmaz temizsin akla zarar"
test_adı = "Ne kadar temizsin"
test_aciklamasi = "Temizliğin ölçündüğü bir test"
gender = "Male"
age = 25

test = test_data(image_place= image_place, image_style= image_style, test_sonucu= test_sonucu, test_adı= test_adı, test_aciklamasi= test_aciklamasi, gender= gender, age= age)



vertexai.init(project="presalesdemo2024", location="us-central1")

prompt = f"Draw a realistic image for this test. Do not add text. It should represent the test result. It should look like a stock image. Only 1 person should be in the image {test.model_dump()}"

generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")

images = generation_model.generate_images(
    prompt = prompt,
    number_of_images = 1,
    aspect_ratio = "16:9",
    negative_prompt = "",
    person_generation = "allow_all",
    safety_filter_level = "block_few",
    add_watermark = True,
)

print(type(images[0]))

output_file = "output.jpg"
images[0].save(location=output_file, include_generation_parameters=False)