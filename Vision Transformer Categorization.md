## Vision Transformer Categorization

| **Category**                                             | **Description**                                              | **Famous Models**                                            |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Encoder-Only (Image Understanding)**                   | Image classification, detection, feature extraction          | ViT, DeiT, Swin, BEiT, ConvNeXt, Twins, Focal Transformer    |
| **Decoder-Only (Image Generation)**                      | Autoregressive or masked token image generation              | MaskGIT, DALL·E (partially), CogView, CogView2               |
| **Encoder-Decoder (Segmentation, Vision-Language, I2I)** | Combines vision encoding with decoding for segmentation, captioning, VQA | SETR, Segmenter, Pix2Seq, BLIP, Flamingo, SAM (Segment Anything Model) |





 ## Encoder-Only Models (Image Understanding)

Used for image classification, feature extraction, and other vision tasks that require comprehension of visual input.

| **Model**                                         | **Description**                                              |
| ------------------------------------------------- | ------------------------------------------------------------ |
| **ViT (Vision Transformer)**                      | The original vision transformer by Google.                   |
| **DeiT (Data-efficient Image Transformer)**       | ViT variant by Facebook optimized for smaller datasets.      |
| **Swin Transformer**                              | Hierarchical transformer with shifting windows, great for detection and segmentation. |
| **BEiT (BERT Pretraining of Image Transformers)** | Inspired by BERT, uses masked image modeling.                |
| **ConvNeXt**                                      | Combines convolutional ideas with transformer blocks.        |
| **Twins**                                         | Combines local and global attention.                         |
| **Focal Transformer**                             | Uses coarse-to-fine attention for better efficiency.         |



## Decoder-Only Models (Image Generation)

Less common, but some models adopt decoder-style structures to autoregressively generate images.

| **Model**              | **Description**                                              |
| ---------------------- | ------------------------------------------------------------ |
| **MaskGIT**            | Generative model using masked token modeling.                |
| **DALL·E (partially)** | Decoder-style transformer used in autoregressive image generation. |
| **CogView / CogView2** | Chinese DALL·E-like models that use decoder transformers for image generation. |



## Encoder-Decoder Models (Image-to-Image / Vision-Language Tasks)

Used for segmentation, captioning, image translation, and multimodal applications.

| **Model**                           | **Description**                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| **SETR (SEgmentation TRansformer)** | Applies encoder-decoder architecture for segmentation.       |
| **Segmenter**                       | ViT-based model for semantic segmentation.                   |
| **Pix2Seq / Pix2Seq v2**            | Treats object detection as a language modeling problem.      |
| **BLIP / BLIP-2**                   | Vision-language models for captioning and VQA.               |
| **Flamingo**                        | Multimodal model with frozen vision encoder and learnable language decoder. |
| **SAM (Segment Anything Model)**    | Promptable, Zero-shot, High-resolution capable Segmentation and Object Detection Model. Uses a Vision Transformer (ViT) backbone for image encoding, and a lightweight decoder that processes prompts (points, boxes, masks) to generate segmentation masks. |