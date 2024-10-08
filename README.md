# DeeperNet: A Deep Learning Educational Framework

Welcome to DeeperNet developed by Eric Marchand, an educational deep learning framework designed to facilitate understanding and experimenting with neural networks (only CPU processing, no GPU/CUDA support). Deeper Net combines the power of Java 17 and JavaFX to offer a hands-on learning experience through interactive training interfaces and visual feedback.

![GUI Preview](src/main/resources/gui.png)

## Features

- **Interactive Training Interface:** Built with JavaFX, Deeper Net provides a user-friendly platform for training neural networks, allowing users to visually track their experiment results in real-time.
- **Easy scale:** The deeper net allows easy scaling of hidden layers, offering a more testable development experience.
![Scaling Preview](src/main/resources/easy-scale.png)
- **Excel Integration:** Deeper Net supports the initiation and configuration of neural networks directly from Excel sheets (`/RN/Samples.xls` or `.numbers`), making it easy to manipulate network parameters and input data.
![XLS Preview](src/main/resources/xls.png)
- **Layer Visualizer** If your neurons are of type PixelNode in a layer you can easily view the layer as an image.
![Layer Preview](src/main/resources/layer-visualizer.png)
- **Implement your own visualizers** Take a look at this implementation of the SIFT vision algorithm.
![Easy integration Preview](src/main/resources/vision-sift.png)
- **TestNetwork.java:** A dedicated launcher that reads network configurations from an Excel sheet and initiates the graphical interface `ViewerFX.java` for visualization.
- **Powerful deep learning PDFs** /resources/ML-Notes.pdf and gradients-histogrammes.pdf.

## Getting Started

### Prerequisites

- Java 17 or higher
- An IDE that supports Java, such as IntelliJ IDEA, Eclipse, or NetBeans
- Basic understanding of neural networks and deep learning concepts

### Setting Up

1. **Clone the repository:**

```bash
git clone https://github.com/stakepoolplace/deepnet.git
```

2. **Open the project in your Java IDE** and make sure it's configured to use Java 17.

3. **Navigate to the `/src/main/resources` directory** to find the sample Excel sheets for network configuration.

### Training a Neural Network

There are two main ways to train a neural network using Deeper Net:

#### Using the API programmatically

```
// Initialisation du réseau
Network network = Network.getInstance().setName("SimpleFFNetwork");

// Configuration des couches
int inputSize = 2; // Taille de l'entrée
int hiddenSize = 2; // Nombre de neurones dans la couche cachée
int outputSize = 1; // Taille de la sortie
network.addLayer(new Layer("InputLayer", inputSize));
network.addLayer(new Layer("HiddenLayer1", hiddenSize));
network.addLayer(new Layer("OutputLayer", outputSize));

// Création des connexions entre les couches
// Note: Implémentez la logique de connexion dans vos classes Layer/Node
network.getFirstLayer().getArea(0).configureLinkage(ELinkage.ONE_TO_ONE, null, false).configureNode(false,
    EActivation.IDENTITY, ENodeType.REGULAR).createNodes(inputSize);

network.getLayer(1).getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true).configureNode(true,
    EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(hiddenSize);

network.getLastLayer().getArea(0).configureLinkage(ELinkage.MANY_TO_MANY, null, true)
    .configureNode(true, EActivation.SYGMOID_0_1, ENodeType.REGULAR).createNodes(outputSize);			

network.finalizeConnections();
```

See example src/test/RN/SimpleFeedforwardNetwork.java

#### Using an Excel Sheet

1. Open the sample Excel sheet in `src/main/resources/Sample.xls` or `.numbers` and configure your network parameters and input data.
2. Save your changes and close the Excel sheet.
3. Run `TestNetwork.java` from your IDE, which will read the Excel sheet and launch `ViewerFX.java` with the specified network configuration.
4. The JavaFX interface will open, allowing you to start and visualize the training process.

#### Documentation


[Documentation via DeeperNet Advisor on chatGPT](https://chat.openai.com/g/g-zAzeIv8Ha-deepernet-advisor)


#### Next steps 
1. Integration of Transformers


## Contributing

We welcome contributions from the community, whether it's improving the documentation, adding new features, or reporting bugs. Please feel free to fork the repository and submit pull requests.

## License

DeeperNet is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Created by Eric Marchand, this project was created as an educational tool for students and enthusiasts wanting to dive deeper into the world of deep learning and neural networks. Special thanks to all contributors and users for their support and feedback.

Happy learning!
