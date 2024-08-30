from typing import Union

from fastapi import FastAPI
from run import createObject
from pydantic import BaseModel, Field
from typing import List
from types import SimpleNamespace
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import uuid
import os


app = FastAPI()
client = OpenAI(
    api_key="sk-GxsUqXrQjRnUZt7QNAOf51kmAGq6GGL6Pi-cmEmlaRT3BlbkFJv4NLo6iMC20ETFG8FGLisQO_WIRLBCzk9tU9uaMIYA"
)

client.images.generate(
  model="dall-e-3",
  prompt="A cute baby sea otter",
  n=1,
  size="1024x1024"
)

class RequestBodyModel(BaseModel):
    image: List[str]
    device: str = Field(default="cuda:0", description="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'")
    pretrained_model_name_or_path: str = Field(default="stabilityai/TripoSR", description="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'")
    chunk_size: int = Field(default=8192, description="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192")
    mc_resolution: int = Field(default=256, description="Marching cubes grid resolution. Default: 256")
    no_remove_bg: bool = Field(default=False, description="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false")
    foreground_ratio: float = Field(default=0.85, description="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85")
    output_dir: str = Field(default="output/", description="Output directory to save the results. Default: 'output/'")
    model_save_format: str = Field(default="obj", description="Format to save the extracted mesh. Default: 'obj'", choices=["obj", "glb"])
    bake_texture: bool = Field(default=False, description="Bake a texture atlas for the extracted mesh, instead of vertex colors")
    texture_resolution: int = Field(default=2048, description="Texture atlas resolution, only useful with --bake-texture. Default: 2048")
    render: bool = Field(default=False, description="If specified, save a NeRF-rendered video. Default: false")

class TextToImageBodyModel(BaseModel):
    prompt: str
    
class ResponseModel(BaseModel):
    status: str
    data: Union[str, List[str]]
    message: str

@app.get("/")
def read_root():
    return {"Hello": "data"}

@app.post("/process")
def process_request(request_body: RequestBodyModel):
    request_body.image = ["examples/pineapple.png","examples/robot.png"]
    createObject(request_body)
    return {"status": "success", "received_data": request_body}

@app.post("/text23d")
def textToImage(request_body: TextToImageBodyModel) -> ResponseModel:
    try:
        # Synchronous call to OpenAI API
        image_response = client.images.generate(
                model="dall-e-3",
                prompt=request_body.prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
                )

        image_data = image_response.data[0].b64_json
        decoded_image = base64.b64decode(image_data)

        # Open the image with PIL
        image = Image.open(BytesIO(decoded_image))

        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        transactionUUID = uuid.uuid4()
        
        unique_filename = f"{transactionUUID}.png"
        image_path = os.path.join('examples', unique_filename)

        # Save the image as a PNG file
        image.save(image_path, "PNG")
        
        args = SimpleNamespace(
            image=[image_path],
            device="cuda:0",
            pretrained_model_name_or_path="stabilityai/TripoSR",
            chunk_size=8192,
            mc_resolution=256,
            no_remove_bg=False,
            foreground_ratio=0.85,
            output_dir="output/",
            uuid=str(transactionUUID),
            model_save_format="obj",
            bake_texture=False,
            texture_resolution=2048,
            render=False
        )
        
        createObject(args)
        
        with open(f"output/{str(transactionUUID)}/mesh.obj", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        return {"status": "success", "data": base64_image, "message": "Image generated successfully"}
    except Exception as e:
        return {"status": "error", "data": str(e), "message": "An error occurred"}