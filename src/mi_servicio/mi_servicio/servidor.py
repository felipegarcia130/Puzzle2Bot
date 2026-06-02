import rclpy
from rclpy.node import Node
from mi_interfaces.srv import Saludo

class Servidor(Node):
    def __init__(self):
        super().__init__("servidor_saludo")
        self.create_service(Saludo,"saludar",self.callback)

    def callback(self,request,response):
        response.mensaje = f"Hola, {request.nombre}!"
        return response

def main(args=None):
    rclpy.init(args=args)
    servidor=Servidor()
    rclpy.spin(servidor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
