# Text to Image Generation Project

## Introduction
This project leverages advanced machine learning models to generate images from text prompts. Using the Stable Diffusion model from Stability AI and Hugging Face's GPT-2 for prompt generation, this project aims to create high-quality images based on user input text.

## Libraries Used
- `pathlib`: For handling file paths.
- `tqdm`: For displaying progress bars.
- `torch`: PyTorch library for tensor operations and GPU acceleration.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `diffusers`: For working with diffusion models.
- `transformers`: For working with transformer models like GPT-2.
- `matplotlib.pyplot`: For plotting images.
- `cv2`: For image processing with OpenCV.

## Setup
To set up the environment and install the necessary libraries, run:
```sh
pip install torch pandas numpy diffusers transformers matplotlib opencv-python
```

## Configuration
A configuration class `CFG` is defined to store various settings and parameters:
- `device`: The device to run the model on, typically "cuda" for GPU acceleration.
- `seed`: Seed for random number generation to ensure reproducibility.
- `generator`: A PyTorch generator for deterministic results.
- `image_gen_steps`: Number of inference steps for generating images.
- `image_gen_model_id`: The model ID for the Stable Diffusion model.
- `image_gen_size`: The size of the generated images.
- `image_gen_guidance_scale`: Guidance scale for the Stable Diffusion model.
- `prompt_gen_model_id`: The model ID for the GPT-2 model.
- `prompt_dataset_size`: Size of the dataset for prompt generation.
- `prompt_max_length`: Maximum length of the generated prompts.

## Model Initialization
The Stable Diffusion model is loaded and moved to the specified device (GPU if available):
```python
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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'
)
image_gen_model = image_gen_model.to(CFG.device)
```

## Image Generation Function
A function `generate_image` is defined to generate images from text prompts:
```python
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image
```

## Usage
To generate an image from a text prompt, call the `generate_image` function with the desired prompt and model:
```python
generate_image("tree", image_gen_model)
```

## Example
Here's an example of generating an image with the prompt "tree":
```python
image = generate_image("tree", image_gen_model)
plt.imshow(image)
plt.axis('off')
plt.show()
```

## Conclusion
This project demonstrates how to use state-of-the-art diffusion models and transformer models to generate images from text prompts. The configuration and setup are designed to be flexible, allowing for easy adjustments to parameters and settings.

## References
1. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
2. [Hugging Face Transformers](https://huggingface.co/transformers/)
3. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
4. [Stable Diffusion](https://stability.ai/)

---

This README provides a comprehensive overview of the text-to-image generation project, including setup instructions, configuration details, and usage examples. Feel free to adjust the content to better fit your specific project requirements.