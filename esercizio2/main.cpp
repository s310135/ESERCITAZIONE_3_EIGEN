#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

//decomposizione PALU
Vector2d sistema_PALU (const Matrix2d& A, const Vector2d& b)
	{
		PartialPivLU<Matrix2d> palu(A);
		Vector2d x_p = palu.solve(b); 
		return x_p;
	}

//decomposizione QR
Vector2d sistema_QR (const Matrix2d& A, const Vector2d& b)
	{
		HouseholderQR<Matrix2d> qr(A);
		Vector2d x_qr = qr.solve(b); 	
		return x_qr;
	}

//errore relativo
double errore_relativo_palu (const Vector2d& x_p, const Vector2d& x)
	{
		double err_rel = (x_p - x).norm() / x.norm();
		return err_rel;
	}
	
double errore_relativo_qr (const Vector2d& x_q, const Vector2d& x)
	{
		double err_rel = (x_q - x).norm() / x.norm();
		return err_rel;
	}

int main()
{
	//soluzione che mi aspetto
	Vector2d x;
	x << -1.0e+0, -1.0e+00;
	
	//sistema 1
	Matrix2d A1;
	A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
	
	Vector2d b1;
	b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	cout << fixed << scientific << setprecision (2) << endl;
	
	Vector2d x1_p = sistema_PALU (A1, b1);
	Vector2d x1_q = sistema_QR (A1, b1);
	cout << "x per sistema 1, metodo PALU: " << x1_p << endl;
	double err_rel1_palu = errore_relativo_palu(x, x1_p);
	cout << "errore relativo sistema 1, metodo PALU: " << err_rel1_palu << endl;
	cout << "x per sistema 1, metodo QR: " << x1_q << endl;
	double err_rel1_qr = errore_relativo_qr (x, x1_q);
	cout << "errore relativo sistema 1, metodo QR: " << err_rel1_qr << endl;
	
	//sistema 2
	Matrix2d A2;
	A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
	
	Vector2d b2;
	b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	Vector2d x2_p = sistema_PALU (A2, b2);
	Vector2d x2_q = sistema_QR (A2, b2);
	cout << "x per sistema 2, metodo PALU: " << x2_p << endl;
	double err_rel2_palu = errore_relativo_palu(x, x2_p);
	cout << "errore relativo sistema 2, metodo PALU: " << err_rel2_palu << endl;
	cout << "x per sistema 2, metodo QR: " << x2_q << endl;
	double err_rel2_qr = errore_relativo_qr (x, x2_q);
	cout << "errore relativo sistema 2, metodo QR: " <<err_rel2_qr << endl;
	
	//sistema 3
	Matrix2d A3;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
	
	Vector2d b3;
	b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	Vector2d x3_p = sistema_PALU (A3, b3);
	Vector2d x3_q = sistema_QR (A3, b3);
	cout << "x per sistema 3, metodo PALU: " << x3_p << endl;
	double err_rel3_palu = errore_relativo_palu(x, x3_p);
	cout << "errore relativo sistema 3, metodo PALU: " << err_rel3_palu << endl;
	cout << "x per sistema 3, metodo QR: " << x3_q << endl;
	double err_rel3_qr = errore_relativo_qr (x, x3_q);
	cout << "errore relativo sistema 3, metodo QR: " <<err_rel3_qr << endl;
	
    return 0;
}

