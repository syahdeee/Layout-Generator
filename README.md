# Layout-Generator
The objective of this research is to create a layout generator system that focuses on the placement of components in advertising poster layouts using a transformer-based model (SGTransformer) based on the input layout graph. Layout graph describes the relationship between components. The following is an illustration of the system that has been built.

![Fix Layout Flow](https://github.com/syahdeee/Layout-Generator/assets/100667458/6fead121-acbb-4c56-987d-691060520403)

The input is a layout graph that represents the required components and relationships between components. Then, the placement of components in the layout will be generated. 

| Layout Graph | Layout         |
|--------------|-----------------|
|  **components:**<br/>["title", "logo", "primary image"]<br /><br />**relationships:**<br />[0, "above", 1]<br />[1, "above", 2]<br />   |<img src="https://github.com/syahdeee/Layout-Generator/assets/100667458/9dff1162-a16b-4f69-bf58-28bfaea1b927" width="300">|
|--------------|-----------------|
