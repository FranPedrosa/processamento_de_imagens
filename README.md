# Processamento de magens
Um trabalho final da disciplina SCC0251 - Processamento de Imagens.

Com a missão de identificar tampinhas de garrafas PET, e com um data set fornecido por uma empresa de geladeiras onde um membro do grupo faz estágio, fizemos uso de HSV, retirando o hue, e extraímos seções de cor vermelha (Coca-cola).
**Entrada e Saída:**<br>
<img src=https://raw.githubusercontent.com/FranPedrosa/processamento_de_imagens/master/readme_imgs/2_4.jpg height="500px">
<img src=https://raw.githubusercontent.com/FranPedrosa/processamento_de_imagens/master/readme_imgs/out.png height="500px">
![image](https://user-images.githubusercontent.com/54639674/179136646-0b6a046d-8ff8-4a59-a9c6-49346606fa4f.png)

Ignorando as partes com saturação indesejada:
![image](https://user-images.githubusercontent.com/54639674/179136704-bd6e1618-ef68-48d2-b796-6b1b41a79916.png)

Foi aplicado Opening nas imagens, de modo que pequenos artefatos fossem removidos e as bordas identificadas fossem suavizadas. Adquirindo as bordas desejadas, utilizou-se uma lógica que permitisse identificar transições em uma 8-neighbourHood, permitindo que distintas formas idênticas fossem filtradas e classificadas de acordo com sua distância com seu centro. Com uso da distância de Manhattan, verificou-se que as tampinhas possuíam um perfil similar, e esses parâmetros foram utilizados para treinar um modelo que identificaria tampinhas nesses tipos de imagens.
Vale ressaltar que o valor Manhattan utilizado e o hue foram obtidos por meio de processos empíricos.
Ao fim, foi aplicado ao modelo as demais imagens, e no geral, bons resultados foram obtidos. Há casos em que algumas tampinhas não são identificadas, ou regiões não interessantes foram consideradas tampinhas, mas representam pequena fração da amostra de dados presente.
