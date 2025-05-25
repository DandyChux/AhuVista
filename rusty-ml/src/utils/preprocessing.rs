pub trait Preprocessor {
    type Input;
    type Output;

    /// Fit preprocessing parameters to training data
    fn fit(&mut self, data: &Self::Input);

    /// Apply transformations to input data
    fn transform(&self, data: Self::Input) -> Self::Output;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: Self::Input) -> Self::Output {
        self.fit(&data);
        self.transform(data)
    }
}
