echo "Starting"
echo "25%----------"
for i in $(seq 1 5);
do
  ./exec ./Imagens_de_Teste/25-percent-marker.pgm ./Imagens_de_Teste/25-percent-mask.jpg ./Imagens_de_Teste/res-cuda-25.tiff 0
  ../../Verificador/exec ../../Verificador/Gabarito/MorphRecon/res-cuda-25.tiff ./Imagens_de_Teste/res-cuda-25.tiff
done
echo "50%----------"
for i in $(seq 1 5);
do
  ./exec ./Imagens_de_Teste/50-percent-marker.jpg ./Imagens_de_Teste/50-percent-mask.jpg ./Imagens_de_Teste/res-cuda-50.tiff 0
  ../../Verificador/exec ../../Verificador/Gabarito/MorphRecon/res-cuda-50.tiff ./Imagens_de_Teste/res-cuda-50.tiff
done
echo "75%----------"
for i in $(seq 1 5);
do
  ./exec ./Imagens_de_Teste/75-percent-marker.jpg ./Imagens_de_Teste/75-percent-mask.jpg ./Imagens_de_Teste/res-cuda-75.tiff 0
  ../../Verificador/exec ../../Verificador/Gabarito/MorphRecon/res-cuda-75.tiff ./Imagens_de_Teste/res-cuda-75.tiff
done
echo "100%----------"
for i in $(seq 1 5);
do
  ./exec ./Imagens_de_Teste/100-percent-marker.jpg ./Imagens_de_Teste/100-percent-mask.jpg ./Imagens_de_Teste/res-cuda-100.tiff 0
  ../../Verificador/exec ../../Verificador/Gabarito/MorphRecon/res-cuda-100.tiff ./Imagens_de_Teste/res-cuda-100.tiff
done
echo "Finished"

