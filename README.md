# DeeperNet: A Deep Learning Educational Framework

Welcome to Deeper Net, an educational deep learning framework designed to facilitate understanding and experimenting with neural networks. Deeper Net combines the power of Java 8 and JavaFX to offer a hands-on learning experience through interactive training interfaces and visual feedback.

## Features

- **Interactive Training Interface:** Built with JavaFX, Deeper Net provides a user-friendly platform for training neural networks, allowing users to visually track their experiment results in real-time.
- **Excel Integration:** Deeper Net supports the initiation and configuration of neural networks directly from Excel sheets (`/RN/Sample.xls` or `.numbers`), making it easy to manipulate network parameters and input data.
- **Console Output:** For those who prefer working directly with code, Deeper Net allows for neural network training with results displayed in the console, offering a more traditional development experience.
- **TestNetwork.java:** A dedicated launcher that reads network configurations from an Excel sheet and initiates the graphical interface `ViewerFX.java` for visualization.

## Getting Started

### Prerequisites

- Java 8 or higher
- An IDE that supports Java, such as IntelliJ IDEA, Eclipse, or NetBeans
- Basic understanding of neural networks and deep learning concepts

### Setting Up

1. **Clone the repository:**

\```bash
git clone https://github.com/yourusername/deeper-net.git
\```

2. **Open the project in your Java IDE** and make sure it's configured to use Java 8.

3. **Navigate to the `/RN` directory** to find the sample Excel sheets for network configuration.

### Training a Neural Network

There are two main ways to train a neural network using Deeper Net:

#### Using the JavaFX Interface

1. Launch the `ViewerFX.java` class from your IDE.
2. Use the interface to configure your neural network and start the training process.
3. Monitor the training progress and results directly within the JavaFX interface.

#### Using an Excel Sheet

1. Open the sample Excel sheet in `/RN/Sample.xls` or `.numbers` and configure your network parameters and input data.
2. Save your changes and close the Excel sheet.
3. Run `TestNetwork.java` from your IDE, which will read the Excel sheet and launch `ViewerFX.java` with the specified network configuration.
4. The JavaFX interface will open, allowing you to visualize the training process.

## Contributing

We welcome contributions from the community, whether it's improving the documentation, adding new features, or reporting bugs. Please feel free to fork the repository and submit pull requests.

## License

DeeperNet is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Created by Eric Marchand, this project was created as an educational tool for students and enthusiasts wanting to dive deeper into the world of deep learning and neural networks. Special thanks to all contributors and users for their support and feedback.

Happy learning!
