use burn::tensor::Tensor;
use rand::Rng;
use textplots::{Chart, Plot, Shape};

/// A simple linear regression model: pass_rate = weight * study_hours + bias
#[derive(Debug)]
struct LinearRegressionModel {
    weight: f32,
    bias: f32,
}

impl LinearRegressionModel {
    /// Initialize model with random weight and bias
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weight: rng.gen_range(0.0..1.0),
            bias: rng.gen_range(0.0..1.0),
        }
    }

    /// Forward pass: predict pass rate from study hours
    fn forward(&self, x: f32) -> f32 {
        self.weight * x + self.bias
    }

    /// Compute gradients using Mean Squared Error (MSE)
    fn compute_gradients(&self, xs: &[f32], ys: &[f32]) -> (f32, f32, f32) {
        let n = xs.len() as f32;
        let mut loss = 0.0;
        let mut grad_w = 0.0;
        let mut grad_b = 0.0;

        for (&x, &y_true) in xs.iter().zip(ys.iter()) {
            let y_pred = self.forward(x);
            let error = y_pred - y_true;
            loss += error.powi(2);
            grad_w += 2.0 * error * x;
            grad_b += 2.0 * error;
        }

        (loss / n, grad_w / n, grad_b / n)
    }

    /// Update model parameters using gradient descent
    fn update(&mut self, grad_w: f32, grad_b: f32, lr: f32) {
        self.weight -= lr * grad_w;
        self.bias -= lr * grad_b;
    }
}

fn main() {
    // Hardcoded dataset: study hours (X) and pass rate (%) (Y)
    let study_hours = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let pass_rates = vec![50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0];

    // Initialize model
    let mut model = LinearRegressionModel::new();
    let learning_rate = 0.01;
    let epochs = 500;

    // Training loop
    for epoch in 0..epochs {
        let (loss, grad_w, grad_b) = model.compute_gradients(&study_hours, &pass_rates);
        model.update(grad_w, grad_b, learning_rate);

        if epoch % 50 == 0 {
            println!(
                "Epoch {}: Loss = {:.4}, Weight = {:.4}, Bias = {:.4}",
                epoch, loss, model.weight, model.bias
            );
        }
    }

    // Print trained model parameters
    println!("Trained Model: Weight = {:.4}, Bias = {:.4}", model.weight, model.bias);

    // Predict pass rate for 0 to 10 study hours
    let mut data_points = Vec::new();
    for hours in 0..=10 {
        let hours_f = hours as f32;
        let predicted_pass_rate = model.forward(hours_f);
        data_points.push((hours_f, predicted_pass_rate));
    }

    // Display plot
    println!("Plot of Study Hours vs. Predicted Pass Rate:");
    Chart::default()
        .lineplot(&Shape::Lines(&data_points))
        .display();
}
