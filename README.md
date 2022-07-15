# Processamento de magens
Um trabalho final da disciplina SCC0251 - Processamento de Imagens.

Os valores utilizados no processo de Filtragem com HSV e no cálculo da distância Manhattan aceitável foram adquiridas empiracamente. Ao longo do desenvolvimento do projeto, identificamos que tais valores eram os que mais se adequavam ao nosso objetivo.

Com a missão de identificar tampinhas de garrafas PET, e com um data set fornecido por uma empresa de geladeiras onde um membro do grupo faz estágio, fizemos uso de HSV, retirando o hue, e extraímos seções de cor vermelha (Coca-cola).
![image](https://user-images.githubusercontent.com/54639674/179136646-0b6a046d-8ff8-4a59-a9c6-49346606fa4f.png)

Ignorando as partes com saturação indesejada:<br>
![image](https://user-images.githubusercontent.com/54639674/179136704-bd6e1618-ef68-48d2-b796-6b1b41a79916.png)

Foi aplicado Opening nas imagens, de modo que pequenos artefatos fossem removidos e as bordas identificadas fossem suavizadas. Adquirindo as bordas desejadas, utilizou-se uma lógica que permitisse identificar transições em uma 8-neighbourHood, permitindo que distintas formas idênticas fossem filtradas e classificadas de acordo com sua distância com seu centro.<br>
![image](https://user-images.githubusercontent.com/54639674/179136805-53a3fcbf-2e0d-46cd-9e93-db081ef9357d.png)
![image](https://user-images.githubusercontent.com/54639674/179136834-da23bfc4-bcaa-4592-9105-c6c16dea3bdd.png)
![image](https://user-images.githubusercontent.com/54639674/179136864-87ccd360-c08d-492f-bfa7-2e998b9d277b.png)
![image](https://user-images.githubusercontent.com/54639674/179136900-e111fbe1-b5ed-4604-ad31-b01cdcea3fcd.png)

Com uso da distância de Manhattan, verificou-se que as tampinhas possuíam um perfil similar, e esses parâmetros foram utilizados para treinar um modelo que identificaria tampinhas nesses tipos de imagens.<br>
![image](https://user-images.githubusercontent.com/54639674/179136960-d4954648-4176-451c-a49b-7d3564a8c1b5.png)
<br>Modelo treinado:<br>
![image](https://user-images.githubusercontent.com/54639674/179137128-6bacc314-a517-4359-84d0-90b7c4111c01.png)

Ao fim, foi aplicado ao modelo às demais imagens, e no geral, bons resultados foram obtidos. Tampinhas vermelhas são contornadas com verde, e regiões vermelhas que não são tampinhas são contornadas com azul/roxo. Há casos em que algumas tampinhas não são identificadas, ou regiões não interessantes foram consideradas tampinhas, mas representam pequena fração da amostra de dados presente.
![image](https://user-images.githubusercontent.com/54639674/179137352-9c9a4f91-2a18-4a03-8713-f91b858656d9.png)
![image](https://user-images.githubusercontent.com/54639674/179137396-938cbb7d-5e70-4afb-9080-7c9e791fca2c.png)
![image](https://user-images.githubusercontent.com/54639674/179137427-f46b77c0-5aa5-4e3b-9669-01216c3e4e89.png)
![image](https://user-images.githubusercontent.com/54639674/179137462-a9a4a190-9d6e-4e1d-aaee-f8893a5f5ce9.png)
![image](https://user-images.githubusercontent.com/54639674/179137486-d4b219a9-1f73-4ba4-a6fe-ed08649656d9.png)
![image](https://user-images.githubusercontent.com/54639674/179137516-21c1849c-367d-40b2-80d0-e98423ec3e8f.png)
![image](https://user-images.githubusercontent.com/54639674/179137537-f8a37b1c-8238-46ed-ad3b-e7e1294e79b1.png)

