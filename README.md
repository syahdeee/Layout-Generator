# Layout-Generator
The objective of this research is to create a layout generator system that focuses on the placement of components in advertising poster layouts using a transformer-based model (SGTransformer) based on the input layout graph. Layout graph describes the relationship between components. The following is an illustration of the system that has been built.

![Fix Layout Flow](https://github.com/syahdeee/Layout-Generator/assets/100667458/6fead121-acbb-4c56-987d-691060520403)

The input is a layout graph that represents the required components and relationships between components. Then, the placement of components in the layout will be generated. Here we list some generated layouts using SGTransformer model.

| Layout Graph | Layout         |
|--------------|-----------------|
|  **components:**<br/>["title", "logo", "primary image"]<br /><br />**relationships:**<br />[0, "above", 1]<br />[1, "above", 2]<br />   |<img src="https://github.com/syahdeee/Layout-Generator/assets/100667458/9dff1162-a16b-4f69-bf58-28bfaea1b927" width="150">|
|  **components:**<br/>["title", "primary image", "primary image"]<br /><br />**relationships:**<br />[0, "above", 1]<br />[1, "right of", 2]<br />   |<img src="https://github.com/syahdeee/Layout-Generator/assets/100667458/b0515cf0-08c0-468d-ac7f-dd4d410e77c9" width="150">|
|  **components:**<br/>["title", "subtitle", "primary image"]<br /><br />**relationships:**<br />[0, "above", 1]<br />[1, "right of", 2]<br />   |<img src="[https://github.com/syahdeee/Layout-Generator/assets/100667458/b0515cf0-08c0-468d-ac7f-dd4d410e77c9](https://github.com/syahdeee/Layout-Generator/assets/100667458/5a7f9e71-c6f3-4f69-8e82-1d9e005859c3)" width="150">|

The dataset we provide is dummy data (dummy_data.json). Dataset used in json format.

