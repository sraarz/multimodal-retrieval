# Multimodal Image Retrieval and Captioning

Content-based image retrieval over a clothing catalogue using vision-language
models, plus an image captioning pipeline trained from scratch.

## Notebooks

### [`TShirt-Retrieval-assignment.ipynb`](notebooks/TShirt-Retrieval-assignment.ipynb)
A retrieval system over a t-shirt catalogue supporting two query modes:

- **Image → image**: CLIP (`openai/clip-vit-base-patch32`) image embeddings ranked
  by cosine similarity
- **Text → image**: free-text queries embedded into the same space; also compared
  against a caption-based route where BLIP generates captions for every catalogue
  image, which are then embedded with a SentenceTransformer and matched against
  the query text

Structured as reusable functions (`initialize_clip`, `initialize_blip`,
`retrieve_images`, `retrieve_images_by_text`, `generate_captions`,
`visualize_top_images`) behind a `main()` entry point. Top-k results are rendered
as image grids for qualitative comparison of the two retrieval routes.

### [`imagecaptioning1.ipynb`](notebooks/imagecaptioning1.ipynb)
An encoder–decoder captioning model built in PyTorch Lightning: a CNN image encoder
feeding an RNN decoder, with a spaCy-tokenized vocabulary, frequency thresholding,
and a custom `Dataset`/collate function handling variable-length caption padding.

## Running

```bash
pip install torch torchvision transformers sentence-transformers scikit-learn \
            pytorch-lightning spacy opencv-python matplotlib pandas
python -m spacy download en_core_web_sm
```

The retrieval notebook expects a directory of catalogue images and an optional
captions CSV; paths are set in the configuration cell. Large embedded outputs were
cleared to keep the notebooks renderable on GitHub.

## Authors

- Sara Rezaei
- Alireza Nejati

Joint project for an applied data mining course.
