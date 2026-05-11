import { useMutation, useQueryClient } from "@tanstack/react-query";
import { todoApi } from "./todoApi";

export const useUpdateTodo = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      todoApi.updateTodo(id, data),

    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },
  });
};