# RB-C1000: The Classification of Real and Bogus Transients using Active Learning and Semi-supervised Learning

The classification of real and bogus transients is a fundamental component in a real-time data processing system and is critical to enable rapid follow-up observations. Most existing methods (supervised learning) require sufficiently large training samples with corresponding labels, which involve costly human labeling and are challenging in the early stage of a time-domain survey. Different from most existing approaches which necessitate massive yet expensive annotated data, We aim to leverage training samples with only 1000 labels available to discover real sources that vary in brightness over time in the early stage of the 2.5-meter Wide-Field Survey Telescope (WFST) 6-year survey. The algorithm will be integrated into the WFST pipeline, enabling efficient and effective classification of transients in the early period of a time-domain survey.

### RB-C1000 architecture

<img src="picture/pipeline.png" alt="vis2" style="zoom:30%;" />

Our method follows a three-stage architecture.   
    * **In the Initial Training Stage**, each labeled sample undergoes convolutional neural network processing to train an initial model. Subsequently, domain experts annotate the $K$ most challenging samples, as determined by the initial model's judgments.   
    * **During the Active Learning Stage**, we employ the combined set of $(M+K)$ labeled samples to train an active training model. From this model, we select the top $V$ samples with high-confidence predictions and assign pseudo-labels to them.   
    * **In the Semi-supervised Learning Stage**, we utilize the expanded dataset of $(M+K+V)$ samples to train a semi-supervised training model. This process is repeated for a total of $R$ iterations to obtain the final results.  