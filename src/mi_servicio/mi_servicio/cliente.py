import rclpy
from rclpy.node import Node
from mi_interfaces.srv import Saludo

class Cliente(Node):
    def __init__(self):
        super().__init__("cliente_saludo")
        self.client = self.create_client(Saludo, "saludar")
        while not(self.client.wait_for_service(timeout_sec=1.0)):
            self.get_logger().info("Esperando al servidor...")

    
    def enviar_request(self, nombre):
        req = Saludo.Request()
        req.nombre = nombre
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    cliente=Cliente()
    resultado = cliente.enviar_request("Felipe")
    print(resultado.mensaje)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
