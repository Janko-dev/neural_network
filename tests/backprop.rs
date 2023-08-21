#[cfg(test)]
mod tests {

    use std::error::Error;

    use neural_network::*;

    #[test]
    fn backprop_basic_operators() -> Result<(), Box<dyn Error>> {

        let a = Matrix::from_vec(vec![2., 3.], (1, 2), true);
        let b = Matrix::from_vec(vec![4., 2.], (1, 2), true);

        let c = a.add(&b)?;
        let d = a.sub(&b)?;
        let e = a.mul(&b)?;
        let f = a.div(&b)?;

        let grads_c = c.backward()?;
        let grads_d = d.backward()?;
        let grads_e = e.backward()?;
        let grads_f = f.backward()?;

        assert_eq!(grads_c.get(a.id()).unwrap().data(), &vec![1., 1.]);
        assert_eq!(grads_c.get(b.id()).unwrap().data(), &vec![1., 1.]);

        assert_eq!(grads_d.get(a.id()).unwrap().data(), &vec![1., 1.]);
        assert_eq!(grads_d.get(b.id()).unwrap().data(), &vec![-1., -1.]);
        
        assert_eq!(grads_e.get(a.id()).unwrap().data(), &vec![4., 2.]);
        assert_eq!(grads_e.get(b.id()).unwrap().data(), &vec![2., 3.]);
        
        assert_eq!(grads_f.get(a.id()).unwrap().data(), &vec![1./4., 1./2.]);
        assert_eq!(grads_f.get(b.id()).unwrap().data(), &vec![-2./4.0_f32.powf(2.), -3./2.0_f32.powf(2.)]);

        Ok(())
    }
    
}