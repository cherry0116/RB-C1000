# RB-C1000: The Classification of Real and Bogus Transients using Active Learning and Semi-supervised Learning

The classification of real and bogus transients is a fundamental component in a real-time data processing system and is critical to enable rapid follow-up observations. Most existing methods (supervised learning) require sufficiently large training samples with corresponding labels, which involve costly human labeling and are challenging in the early stage of a time-domain survey. Different from most existing approaches which necessitate massive yet expensive annotated data, We aim to leverage training samples with only 1000 labels available to discover real sources that vary in brightness over time in the early stage of [the 2.5-meter Wide-Field Survey Telescope (WFST) 6-year survey](https://arxiv.org/abs/2306.07590). The algorithm will be integrated into the WFST pipeline, enabling efficient and effective classification of transients in the early period of a time-domain survey.

### RB-C1000 architecture

<img src="picture/pipeline.png" alt="vis2" style="zoom:30%;" />

Our method follows a three-stage architecture:    
- **In the Initial Training Stage**, each labeled sample undergoes convolutional neural network processing to train an initial model. Subsequently, domain experts annotate the $K$ most challenging samples, as determined by the initial model's judgments.   
- **During the Active Learning Stage**, we employ the combined set of $(M+K)$ labeled samples to train an active training model. From this model, we select the top $V$ samples with high-confidence predictions and assign pseudo-labels to them.   
- **In the Semi-supervised Learning Stage**, we utilize the expanded dataset of $(M+K+V)$ samples to train a semi-supervised training model. This process is repeated for a total of $R$ iterations to obtain the final results.  

### Dataset

We have constructed new real/bogus classification datasets from [the Zwicky Transient Facility (ZTF)](https://arxiv.org/abs/1902.01932) to verify the effectiveness of our approach. We collected three newly compiled datasets for the real/bogus classification task, including two single-band datasets (ZTF-NEWg: g-band, ZTF-NEWr: r-band) and one mixed-band dataset (ZTF-NEWm: take half of the g-band data and half of the r-band data). Each of the three dataset with 13000 real sources and 30000 bogus detections. It can be downloaded at [here]().

## Usage
### Train

Clone this project and then go to the code directory:

    git clone https://github.com/cherry0116/RB-C1000.git
    cd cls_code

We train the model on the following environments:

    Python 3.8
    Pytorch 1.13.1
    Torchvision 0.14.1 
    Torchaudio 0.13.1
    Cuda 11.6

You can build the environment quickly by installing the requirements:

    conda env create -f requirements.yml
    conda activate astro_cls

Train RB-C1000:

### Evaluate