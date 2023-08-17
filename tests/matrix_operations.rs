#[cfg(test)]
mod tests {
    use std::error::Error;

    use neural_network::*;

    #[test]
    fn matrix_multiplication() -> Result<(), Box<dyn Error>>{
        let a = Matrix::from_vec(vec![1., 2.], (1, 2));    
        let b = Matrix::from_vec(vec![2., 3.], (2, 1));
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), (1, 1));
        assert_eq!(c.data(), &vec![8.]);

        let a = Matrix::from_vec(vec![1., 2., 3., 4.], (2, 2));    
        let b = Matrix::from_vec(vec![2., 3.], (2, 1));
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), (2, 1));
        assert_eq!(c.data(), &vec![8., 18.]);

        let a = Matrix::from_vec(vec![1., 2.], (1, 2));    
        let b = Matrix::from_vec(vec![2., 3., 4., 5.], (2, 2));
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), (1, 2));
        assert_eq!(c.data(), &vec![10., 13.]);

        let a = Matrix::from_vec(vec![1., 2., 0., 1.], (2, 2));    
        let b = Matrix::from_vec(vec![2., 5., 1., 6., 7., 1.], (2, 3));
        let c = a.matmul(&b)?;
        assert_eq!(c.shape(), (2, 3));
        assert_eq!(c.data(), &vec![14., 19., 3., 6., 7., 1.]);

        let a = Matrix::from_vec(vec![1., 2.], (1, 2));    
        let b = Matrix::from_vec(vec![2., 3.], (2, 1));
        let c = a.matmul(&b)?.matmul(&a)?;
        //  [1 2] @ [2] = [8] @ [1 2] = [8 16]
        //          [3]
        assert_eq!(c.shape(), (1, 2));
        assert_eq!(c.data(), &vec![8., 16.]);

        let a = Matrix::randn(0., 1., (2, 3));    
        let b = Matrix::randn(0., 1., (2, 3));
        let res = a.matmul(&b);
        assert!(res.is_err());

        Ok(())
    }


    #[test]
    fn matrix_element_wise_addition() -> Result<(), Box<dyn Error>> {
        let a = Matrix::from_vec(vec![1., 2., -2., 1.], (2, 2));    
        let b = Matrix::from_vec(vec![2., -5., 1., 6.], (2, 2));
        let c = a.add(&b)?;
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.data(), &vec![3., -3., -1., 7.]);

        let b = Matrix::randn(0., 1., (4, 4));
        let res = a.add(&b);
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn matrix_element_wise_multiplication() -> Result<(), Box<dyn Error>> {
        let a = Matrix::from_vec(vec![1., 2., -2., 1.], (2, 2));    
        let b = Matrix::from_vec(vec![2., -5., 1., 6.], (2, 2));
        let c = a.mul(&b)?;
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.data(), &vec![2., -10., -2., 6.]);

        let b = Matrix::randn(0., 1., (4, 4));
        let res = a.mul(&b);
        assert!(res.is_err());
        
        Ok(())
    }

    #[test]
    fn matrix_transpose() {
        let a = Matrix::from_vec(vec![1., 2., -2., 1., 4., 6.], (2, 3));
        let c = a.t().t();
        
        assert_eq!(c.data(), a.data());
    }
}