import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()


def createObject(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    timer.start("Initializing model")
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    timer.end("Initializing model")

    timer.start("Processing images")
    images = []

    if args.no_remove_bg:
        rembg_session = None
    else:
        rembg_session = rembg.new_session()

    for i, image_path in enumerate(args.image):
        if args.no_remove_bg:
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image = remove_background(Image.open(image_path), rembg_session)
            image = resize_foreground(image, args.foreground_ratio)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            if not os.path.exists(os.path.join(output_dir, args.uuid)):
                os.makedirs(os.path.join(output_dir, args.uuid))
            image.save(os.path.join(output_dir, args.uuid, f"input.png"))
        images.append(image)
    timer.end("Processing images")

    for i, image in enumerate(images):
        logging.info(f"Running image {i + 1}/{len(images)} ...")

        timer.start("Running model")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        timer.end("Running model")

        if args.render:
            timer.start("Rendering")
            render_images = model.render(scene_codes, n_views=30, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(os.path.join(output_dir, args.uuid, f"render_{ri:03d}.png"))
            save_video(
                render_images[0], os.path.join(output_dir, args.uuid, f"render.mp4"), fps=30
            )
            timer.end("Rendering")

        timer.start("Extracting mesh")
        meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
        timer.end("Extracting mesh")

        out_mesh_path = os.path.join(output_dir, args.uuid, f"mesh.{args.model_save_format}")
        if args.bake_texture:
            out_texture_path = os.path.join(output_dir, args.uuid, "texture.png")

            timer.start("Baking texture")
            bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
            timer.end("Baking texture")

            timer.start("Exporting mesh and texture")
            xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
            Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
            timer.end("Exporting mesh and texture")
        else:
            timer.start("Exporting mesh")
            meshes[0].export(out_mesh_path)
            timer.end("Exporting mesh")