import matplotlib.pyplot as plt
import cv2
import os

class LegoAnalise:

    def ler_imagem(caminho):
        imagem = cv2.imread(caminho)
        return imagem

    def binarizacao_sem_fundo(img):
        img = img[:,:,2]
        _, bin = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
        return bin

    def erodir(img, kernel_size=(33,33)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img_erodida = cv2.erode(img, kernel)
        return img_erodida

    def fechar_falhas(img, kernel_size=(10,10)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img_fechada = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img_fechada

    def reconhecer_e_desenhar(img_bin, img_orig):
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lego_infos = []
        
        for i, contorno in enumerate(contours):
            area = cv2.contourArea(contorno)
            
            if area < 1000:
                continue
            
            perimetro = cv2.arcLength(contorno, True)
            aproximacao = cv2.approxPolyDP(contorno, 0.05 * perimetro, True)
            
            e_retangulo = len(aproximacao) == 4
            
            rect = cv2.minAreaRect(contorno)
            _, (w, h), _ = rect
            
            if w < h:
                w, h = h, w
                
            proporcao = w / h if h != 0 else 0
            
            if e_retangulo:
                if (2000 < area < 4000) and (proporcao < 1.10):
                    tamanho_lego = "(2x2)"
                elif (6000 < area < 10000) and (2 < proporcao < 4):
                    tamanho_lego = "(2x4)"
                elif (11000 < area < 16500 and (3.5 < proporcao < 5)):
                    tamanho_lego = "(2x6)"
                elif (14000 < area < 20000 and (proporcao >= 5)):
                    tamanho_lego = "(2x8)"
                else:
                    tamanho_lego = "Not Identified"
            else:
                tamanho_lego = "Irregular Piece"

            lego_infos.append({
                'tamanho_lego': tamanho_lego,
                'area': area,
                'contorno': contorno,
                'proporcao': proporcao
            })
        
        lego_infos.sort(key=lambda x: x['area'], reverse=True)
        
        return_info = f'Legos Caracterizados por Tipo de Peça:\n'
        for i, shape in enumerate(lego_infos, 1):
            return_info += f"{i}. Tipo: {shape['tamanho_lego']}\n"
            return_info += f" Área: {shape['area']:.2f} pixels\n"
            return_info += f" Proporção (Largura/Altura): {shape['proporcao']:.2f}\n\n"
        
        output_image = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        for i, shape in enumerate(lego_infos):

            color = (0, 0, 255)

            cv2.drawContours(output_image, [shape['contorno']], -1, color, 3)
            
            M = cv2.moments(shape['contorno'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(output_image, f"{i+1}. {shape['tamanho_lego']} ", (cx-20, cy-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return return_info, output_image
    
    def resolver(caminho_imagem):
        img=LegoAnalise.ler_imagem(caminho_imagem)
        bin=LegoAnalise.binarizacao_sem_fundo(img)
        bin=LegoAnalise.fechar_falhas(bin)
        bin=LegoAnalise.erodir(bin)
        infos, res=LegoAnalise.reconhecer_e_desenhar(bin, img)
        return infos, res

    def display_all(caminho_pasta, juntas=True):
        imagens = []
        for ficheiro in os.listdir(caminho_pasta):
            if ficheiro.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagens.append(os.path.join(caminho_pasta, ficheiro))

        if juntas:
            _, axes = plt.subplots(2, len(imagens), figsize=(18, 4))
            axes=axes.ravel()

            for i in range(len(imagens)):
                imagem=cv2.cvtColor(LegoAnalise.ler_imagem(imagens[i]), cv2.COLOR_BGR2RGB)
                axes[i].imshow(imagem)
                axes[i].axis('off')

            for i in range(len(imagens)):
                infos, res = LegoAnalise.resolver(imagens[i])
                print('Imagem ', i+1, ':')
                print(infos)
                axes[len(imagens)+i].imshow(res, cmap='binary')
                axes[len(imagens)+i].axis('off')
            plt.show()
        else:
            for i in range(len(imagens)):
                infos, res = LegoAnalise.resolver(imagens[i])
                print('Imagem ', i+1, ':')
                print(infos, '\n --------------------------------')
                plt.imshow(res)
                plt.axis('off')
                plt.show()

